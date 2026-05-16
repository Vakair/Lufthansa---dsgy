# Aviation Delay Prediction — Dokumentáció

A projektet Databricksben valósítottam meg PySpark + Pandas/Scikit-Learn/XGBoost stack-kel. A notebook reprodukálhatóan végigfut a beolvasástól a kiértékelésig. Az alábbi dokumentáció a feladatkiírás három fő szekcióját (EDA, Modellezés, Business Interpretation) követi.

---

## 1. Exploratory Data Analysis (EDA)

### Hogyan álltam neki

Először a dátumoszlopnál kezdtem, mert az időalapú adatoknál ez az első, ami félre tud csúszni. Ellenőriztem, hogy a `flight_datetime` tényleg dátum-típusként van-e értelmezve (kellett egy `to_timestamp` cast). Ezután lekérdeztem az időtartományt: az adathalmaz **2024-01-01 és 2024-05-04 között 124 napot fed le**, tehát egy tört évet, csak január–május. Ez fontos, mert szezonalitást vagy nyári csúcsforgalmat nem tudunk megtanulni belőle. A `flight_datetime`-ből kinyertem a hónap (`flight_month`) és a hét napja (`flight_dayofweek`) feature-öket, hogy a modell tudjon mintázatot találni a heti ciklikusságban.

Ezután megnéztem **óra-konzisztenciát**: a `flight_datetime` órája 100%-ban egyezett a `scheduled_departure_hour` értékével — tehát a `scheduled_departure_hour` redundáns. Ettől még a kódban benne hagytam, mert a fa-alapú modellek úgyis ignorálják a redundáns oszlopot, és nem akartam fölöslegesen módosítgatni az adatkészletet.

**Duplikátum-vizsgálat**: a `(flight_datetime, origin, destination)` kombinációkra nem volt egyetlen duplikátum sem.

**Havi eloszlás**: a sorok nagyjából egyenletesen oszlanak el a 4-5 hónap között, nincs furcsa lyuk.

### Hiányzó értékek

Két oszlopban volt null érték:
- `visibility` — 7,73%
- `maintenance_events_last_30d` — 12,87%

Mielőtt eldöntöttem volna, mit kezdjek velük, megvizsgáltam, hogy a hiányzás összefügg-e valami máshelyzeti változóval:

- `visibility` hiányokat lebontottam **`origin` szerint** — kb. egyenletesen oszlik szét a reptereken.
- `maintenance_events_last_30d` hiányokat lebontottam **`aircraft_age` és `aircraft_type` szerint** is — itt sem volt jelentős különbség a csoportok között.

Mivel a hiányzás nagyjából véletlennek tűnt (MCAR-nak vettem), és **mindössze 6000 soros az adathalmaz**, semmiképp nem akartam sorokat dobni. Már itt sejtettem, hogy az adatok mennyisége elég komoly korlát lesz, és minden sorra szükség lesz.

Globális átlag helyett **csoportos medián imputációt** alkalmaztam Spark ablakfüggvénnyel:
- `visibility` → az adott `origin` mediánjával
- `maintenance_events_last_30d` → az adott `aircraft_type` mediánjával

A medián mellett szándékosan döntöttem az átlag helyett, hogy az esetleges outlierek (pl. egy szélsőséges időjárású nap) ne torzítsák. A pótlások mellé létrehoztam két flag oszlopot (`visibility_missing_flag`, `maintenance_missing_flag`), így a modell ki tudja használni azt az információt is, hogy egy adott sornál eredetileg hiányzott-e az érték — esetleg maga a "hiányzás ténye" is hordoz valami jelet.

### A célváltozó problémája (ez volt a legkellemetlenebb felfedezés)

Megnéztem a `delay_over_15m` eloszlását: csak **3.97%** volt 1-es. Ez nagyon kevés, és itt biztos voltam benne, hogy lesz dolgom az imbalanced eloszlással.

Aztán gondoltam egyet, és leellenőriztem a címke konzisztenciát az `actual_delay_minutes` alapján: **163 sorban volt hibás a címke** (vagy 15 perc fölött késett és mégis 0-t kapott, vagy fordítva). Ez egyértelműen adathiba, nem véletlen.

Ezért **újradefiniáltam a célváltozót** tiszta logikai szabállyal: `delay_over_15m = 1`, ha `actual_delay_minutes > 15`, különben 0. Így a pozitív arány **5.72%**-ra módosult — még mindig nagyon kiegyensúlyozatlan, de legalább konzisztens.

Az `actual_delay_minutes` eloszlását is megnéztem: nem voltak negatív értékek (korai indulás).

### Data leakage vizsgálat

Ez volt a másik kritikus pont. Több gyanús oszlop is volt, amelyek **a predikció pillanatában nem állnának valójában rendelkezésre**, vagy közvetlenül a célváltozóból származnak. Kiszámoltam a korrelációjukat a célváltozóval, hogy lássam, mi a helyzet:

| Oszlop | Korreláció/oszlop értelmezése | Döntés |
|---|---|---|
| `actual_delay_minutes` | Utólag ismert adat (a target ebből származik) | **DROP** |
| `actual_gate_out_time_diff` | leakage (szintén utólagos esemény) | **DROP** |
| `maintenance_closed_after_pushback` | leakage (pushback után, már elindult a gép vagyis utólagos adat) | **DROP** |
| `final_delay_reason` | a késés OKA, ez is csak utólag ismert | **DROP** |
| `ops_delay_prediction_v2` | alacsony korreláció, de a név alapján gyanús volt | meghagytam (alacsony korr.) |
| `sched_buffer_mins_latest` | alacsony korreláció | meghagytam |
| `previous_leg_delay` | alacsony korreláció, de **logikailag legit** (előző leg-ről jön) | meghagytam |
| `crew_status_new_FINAL` | alacsony korreláció | meghagytam |

Tehát csak azokat dobtam, amik egyértelmű leakage-ek voltak. A többit a feature importance majd úgyis kiszórja, ha nem hasznos.

### Mintázatok és gyenge jelek

A numerikus oszlopokat binneltem (kategóriákra bontottam), majd csoportonként megnéztem a `delay_over_15m` arányt. Néhány érdekes megfigyelés:

- **Aircraft type / aircraft age**: nem mutatott számottevő eltérést a kategóriák között.
- **Route distance**: gyakorlatilag flat. Nincs egyértelmű jelzés, hogy a hosszabb járatok jobban vagy kevésbé késnének.
- **Visibility / wind / precipitation**: ugyan más kategóriáknál ezek erős jelek szoktak lenni, de itt egyik bin sem mutatott kiugró delay rate-et.
- **Previous leg delay**: itt volt valami jel — ha az előző leg már késett, akkor a jelenlegi is nagyobb valószínűséggel késik. Logikus is, ez a "cascade effect".

**Konklúzió az EDA végén:** egyik feature sem mutatott önmagában erős korrelációt (mind <0.10) a célváltozóval. Itt már sejtettem, hogy a modellezés frusztráló lesz — nincs olyan változó, amire egy modell rátudna kapaszkodni. Plusz csak 6000 sorom van. Plusz a pozitív osztály 5.7%. Ez együtt **rossz előjel** volt a prediktív erő szempontjából, de pont ezt is kéri a feladatkiírás: hogy ne aggódjak a tökéletes eredmény miatt, hanem a gondolkodásmódot mutassam be.

---

## 2. Modellezés

### Preprocessing

1. **PySpark → Pandas konverzió** (`toPandas()`). 6000 sornál a Spark már fölösleges, a Scikit-Learn és XGBoost rugalmasabb. A konverzió előtt a tisztításokat (medián imputáció, leakage drop, célváltozó újradefiniálás) még Sparkban végeztem el.
2. **Időrendi sorba rendezés** — ez azért volt fontos, mert a split alapja az index lesz.
3. **One-Hot Encoding** a kategorikus oszlopokra (`origin`, `destination`, `aircraft_type`), `drop_first=True` paraméterrel a multikollinearitás elkerülésére.
4. **Az eredeti `flight_datetime` oszlopot eldobtam** (a hónap és hét napja már kinyerve).

### Validációs stratégia: időalapú split (70 / 15 / 15)

Szándékosan **NEM** random splitet használtam, hanem szigorúan index szerinti vágást: első 70% = train, következő 15% = validation, utolsó 15% = test. Erre két okból:

1. **Időalapú adatoknál a random split data leakage** — a modell jövőből származó információkat látna a tréning során.
2. A 70/15/15 felosztás **a valós production működést szimulálja**: a modellt egy múltbeli időszakon tanítjuk, és a jövőbeli járatokra alkalmazzuk.

### Választott metrikák

Az imbalanced eloszlás miatt az **Accuracy önmagában megtévesztő**. Ha a modell mindig 0-t mond, ~94% accuracy-t ér el — de teljesen használhatatlan. Ezért a fókusz:

- **Recall** — a valós késések mekkora részét kapja el a modell. Üzletileg ez számít legjobban: ha egy késést nem találunk meg, az drága.
- **PR-AUC (Average Precision)** — az imbalanced eloszlásnál ez a legmegbízhatóbb összegző metrika. A ROC-AUC-nál érzékenyebb a ritka pozitív osztályra.
- Másodlagosan **Precision** és **F1**, valamint **ROC-AUC** (Receiver Operating Characteristic – Area Under Curve) kontextus céljából.

### Baseline modellek (3 db)

A feladat 1 baseline-t kért, de szándékosan **hármat** csináltam, mert mindegyik más-más szempontból tanulságos:

1. **Dummy Classifier** ('strategy='most_frequent''): Ez a naive baseline minden járatra azt jósolja, hogy NEM fog késni (mindig 0). Arra használtam, hogy bebizonyítsam az Accuracy csalókaságát: a modell 93.22%-os pontosságot ér el úgy, hogy a valóságban használhatatlan (Recall és PR-AUC lényegében 0).

2. **Üzleti heurisztika** — egyszerű szabály: ha `previous_leg_delay > 15`, akkor késést jósol. Ez a domain tudást reprezentálja, és kíváncsi voltam, hogy ezt mennyivel veri meg egy ML modell. (nem nagyon.)

3. **Logistic Regression** (`class_weight='balanced'`, `max_iter=2000`) — klasszikus lineáris baseline. A `balanced` class weight kompenzálja az imbalanced eloszlást azzal, hogy a ritka osztály mintáit nagyobb súllyal veszi.

### Fejlettebb modellek (2 db)

Gondolkodtam egy saját kis neurális hálón (GRU vagy LSTM, mivel idősoros az adat), de **6000 sornál egy mélytanulási modell over-engineering**, és garantáltan túltanulna. Maradtam a fa-alapú megközelítésnél, mert ezek robusztusak és kevés tuning mellett is jól szoktak teljesíteni — plusz korábbi projektjeimen már dolgoztam velük.

1. **Random Forest Classifier** (`n_estimators=200`, `max_depth=7`, `class_weight='balanced_subsample'`)
    - A `max_depth=7` korlát a túltanulás ellen.
    - A `balanced_subsample` a fa-szintű mintavételezésnél kezeli az imbalanced eloszlást.

2. **XGBoost Classifier** (`n_estimators=200`, `max_depth=5`, `learning_rate=0.05`, `scale_pos_weight=arány`)
    - A `scale_pos_weight`-et a `negatív / pozitív` arány alapján számoltam (~16), így a modell minden pozitív minta hibájáért ~16x büntetést kap.
    - `eval_metric='aucpr'` — közvetlenül a Precision-Recall görbére optimalizál, ami pont az, ami imbalanced eloszlásnál kell.

### Eredmények — Test halmaz (utolsó 15%)

| Modell | Recall | Precision | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Dummy (mindig 0) | 0.000 | 0.000 | 0.000 | 0.0678 | 0.500 |
| Üzleti szabály | 0.5082 | 0.0829 | 0.1425 | — | — |
| Logistic Regression | 0.3115 | 0.0795 | 0.1267 | 0.0993 | 0.5399 |
| Random Forest | 0.0656 | 0.2222 | 0.1013 | 0.1104 | 0.5462 |
| XGBoost | 0.0820 | 0.1163 | 0.0962 | 0.0912 | 0.5132 |

**Megfigyelések:**

- A fejlettebb modellek (RF, XGBoost) **konzervatívan** működtek: nagyon óvatosan jósoltak késést. Magasabb a Precision, viszont alacsony a Recall (~6-8%). Tehát ha 1-est mondanak, az többször helyes — de a tényleges késések 90%+-át elszalasztják.
- A Logistic Regression és az üzleti szabály **bátrabban riasztott**, magasabb Recall-lal, de cserébe rengeteg false positive-val.
- A PR-AUC értékek **alig haladják meg a Dummy baseline-t (0.068)**. A ROC-AUC értékek 0.5-0.56 között mozognak, ami szinte véletlenszerű.

Ez nem meglepetés: az EDA-ban már láttam, hogy nincs erős prediktív jel az adatban. A számok ezt megerősítik.
A részletes vizualizációkat (ROC görbék, PR görbék, confusion matrix-ok, feature importance) a notebookban találod.

---

## 3. Business Interpretation

### Mi okozza leginkább a késéseket?

**Őszintén: nincs egyetlen kiugróan mérhető tényező** az adatok alapján. A Random Forest feature importance diagramja szerint a top változók nagyjából sorrendben:
1. `ops_delay_prediction_v2` (belső operatív becslés)
2. `airport_congestion_index` (repülőtéri leterheltség)
3. `previous_leg_delay` (előző leg-ről áthozott késés)
4. `turnaround_minutes` (fordulóidő)
5. `runway_utilization` (kifutó kihasználtság)

Az **időjárási tényezők (szél, csapadék, visibility) elhanyagolható szerepet játszottak**, ami furcsa az aviation kontextusban — valódi adatnál ez egészen biztosan nem így lenne, de itt szintetikus adatról van szó.

A legtöbb fontos változó valamilyen **ütemezési vagy operatív feszültséget** mér, nem pedig külső tényezőt.

### Mely tényezők befolyásolhatók operációs oldalról?

A feature importance szerint (fenntartva, hogy a modell gyenge prediktív erejű):

- **`turnaround_minutes`** — közvetlenül befolyásolható: hosszabb pufferidő = kevesebb cascade késés.
- **`airport_congestion_index`** — slot-optimalizálással és reptér-koordinációval részben kezelhető.
- **`previous_leg_delay`** — közvetve: ha az előző leg-eknél is jobb a teljesítmény, ez is javul.
- **`runway_utilization`** — pálya-allokáció optimalizálása.

Üzleti szempontból a **fordulóidő-puffer növelése** lenne a legközvetlenebb tőke, ami csökkenthetné a továbbgyűrűző (cascade) késéseket. Persze ez költséges, mert a kihasználtság rovására megy.

### Mennyire bíznál a modellben?

**Semennyire, jelenleg.** A PR-AUC értékek rendkívül alacsonyak, a ROC-AUC közel a random szinthez. A Random Forest a tényleges késések 93-94%-át elszalasztja a test halmazon. Talán **ha egy tízszer ekkora, valódi operációs adathalmazon** lehetne tanítani — több hónapot/évet lefedő, történelmi mintákkal — akkor lenne értelme bízni benne. Most nincs.

### Milyen limitációi vannak?

A modellnek **jelenleg szinte csak limitációi vannak**:

- **Adatmennyiség**: 6000 sor egyszerűen nem elég ilyen ritka pozitív osztálynál (~340 pozitív minta összesen).
- **Imbalanced eloszlás**: 5.72% pozitív arány. A modell hajlamos a többségi osztály felé tolódni.
- **Gyenge jelek**: egyetlen feature sem korrelál erősen (>0.10) a célváltozóval.
- **Időtáv**: 4 hónap (jan-máj), nincs szezonalitás, nincs nyári csúcsforgalom.
- **Adatminőség**: a célváltozó eredetileg 163 hibás címkét tartalmazott (ezt javítottam).
- **Szintetikus adat**: a feladatkiírás szerint mesterségesen generált, így az összefüggések sem feltétlenül valóságosak.

Egyetlen pozitívum: **a semminél jobb**. Néhány valódi True Positive-ot megfogott, tehát nem teljesen vak.

### Hogyan használnád production környezetben?

Ebben a formában én **nem vezetném be élesben**. De ha mindenáron kellene, akkor kizárólag **döntéstámogató "jósgömbként"**:

- A diszpécser képernyőjén, ha a modell késést jelez, csak ennyit írnék ki: **"LEHETSÉGES 15 PERC FELETTI KÉSÉS"** — jelezve, hogy az információ irányadó, de nem biztos.
- Azt a kijelentést, hogy egy járat **"NEM FOG KÉSNI"**, teljesen letiltanám. Az alacsony Recall miatt a modell rengeteg valós késést elszalaszt, ezért egy ilyen kijelentés félrevezető lenne.

**Production rollout előtt** mindenképp kellene:
1. **Sokkal több adat** (legalább 1-2 év, valós járatokról).
2. **Folyamatos monitoring** a model driftre — különösen szezonális hatások miatt.
3. **A/B tesztelés** vs. a jelenlegi heurisztika (akár csak a `previous_leg_delay > 15` szabály ellen).
4. **Threshold tuning** üzleti költség alapján: mennyibe kerül egy false alarm vs. egy missed delay? Ez alapján kell beállítani a döntési küszöböt.

---

## Reprodukálhatóság

- A notebook elejétől a végéig sorrendben végigfut, egyedül az excel file lokációját kell átírni.
- Az időalapú split miatt a train/val/test felosztás determinisztikus.
- A medián imputáció és One-Hot Encoding determinisztikus.
- Külső függőség: `xgboost` (`pip install xgboost` cellában a notebook elején).
