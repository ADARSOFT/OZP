1. Ucitati skup podataka koji je dodeljen vasem timu - odradjeno
2. Prikazati deskritivnu statistiku skupa podataka (tipovi podataka, rasponi vrednosti, nedostajuce vrednosti)
	- podelio sam podatke na numericke, kategoricke i output labelu
	- koristeci funkciju unique() prikazao sam koliko ima jedinstvenih vrednosti u koloni i koliko se puta ponavljaju
	- koristeci funkciju value_counts(), uvideo sam da postoji u koloni nr.employeed zapis (780 index), koji ima nedozvoljenu vrednost 'no' i ta vrednost menja tip kolone.
	obrisao sam taj red i vratio kolonu na numeric tip. 
	- funkcijom info() prikazao sam tipove podataka za kolone u data setu
	- funkcijom describe() prikazao sam neke od statistickih mera (ukupan broj, srednju vrednost, std, min, max, kvartile itd.)
	- funkcijom isnull() proverio sam da li u redovima i kolonama ima nedostjucih podataka
3. Pripremiti skup podataka tako da bude pogodan za prediktivno podelovanje
	- feature selection (to do) ovo mozda ide na kraju
	- Nedostajuce numericke podatke sam popunio funkcijom fillna() kojoj sam prosledio srednje vrednosti kolona 
	- Prikazao sam procenat 'undefined' vrednosti po koloni za kategoricke podatke
	- Prikazao sam procenat Null vrednosti po koloni za kategoricke podatke
	- Prema nedostajucim kategorickim podacima Null ili Unknown sam pristupio sa 3 razlicita aspekta:
		1. Kolonu Marital (kolona sa najmanje nedostajucih vrednosti) sam popunio sa najfrekfentnijom vrednoscu (koristio sam fuknciju mode()). 
		U cilju da izbegnem smanjivanje varijance podataka izabrao sam nju jer ima najmanji procenat nedostajucih podataka.
		2. Zapazio sam da kolone HOUSING I LOAN imaju visoku korelaciju nedostajucih vrednosti (indexi nedostajucih redova su isti). Odlucio sam da koristim prediktivni model 
		kako bih dobio vrednosti koje nedostaju za obe kolone. Za ovaj zadatak sam koristio Logisticku regresiju kao klasifikator. Numericke kolone sam koristio kao featur-e 
		a labela odnosno output klasa mi je bila kolona sa nedostajucim podacima. Podaci za ucenje su mi bili zapisi ovih kolona koji su imali vrednosti a ostale sam predvideo.
		3. Nedostajuce vrednosti 'Unknown' kolone DEFAULT odlucio sam da posmatram kao novu kategoriju iz razloga sto ih ima skoro 20% procenata, pa sam smatrao da bi imputacija
		pogorsala rezultate. Kako bih dosao do konkretnog zakljucka mogao bih da napravim eksperiment i da koristim neku drugu taktiku i da uporedim rezultate. 
	- Pretvaranje podataka iz kategorickih u numericke sam zavrsio pomocu tehnike dummy coding, funkcijom get_dummies(). Koristio sam tehniku dummy coding ispred Label encoding
	kako ne bi doslo do pogresnih zakljucaka u linearnim modelima.
	- Resavanje problema sa ekstremnim vrednostima. Koristio sam tehniku zscore. Prikazao sam zapise koji imaju vise od 3 standardne devijacije. Obrisao sam jedan red za koji sam 
	smatrao da je outlier, ostale sam sacuvao. Gledajuci Distribucije tih kolona, zakljucio sam da nisu outlier-i vec da je distribucija podataka takva. Takodje sam prikazao box-plot
	tehniku za nekoliko kolona. 
	- Prikazao sam matricu korelacije i pairplot dijagram 
	- Prikazao sam bar dijagram kako bih prikazao broj outlier-a po kolonama
	
4. Podeliti skup podataka na trening i test u odnosu 70:30

5. Analizirajte glavne komponente skupa podataka
	- Analizirane su glavne komponente na 98% kumulativne varijanse i na 100%
6. Kreirajte klaster model i odredite klastere svake instance. Karakterisite dobijene klastere
7. Kreirajte minimalno 3 prediktivna modela (sa default parametrima), uporedite ih cross validacijom i ocenite gresku na test setu (minimum 2 mere evaluacije).
Koristite pipeline.
8. Promenite minimalno 2 parametra kod najboljeg modela i ocenite gresku na test setu.
9. Testirajte prediktivne modele na atributima koji kumulativno nose 98% varijanse.
	- varijanta 1 (PCA)
	- Pomocu ExtraTreesClassifier-a i propertija feature_importances_ dobio sam feature importance
10. Podelite inicijalni skup po klasterima koje ste dobili i sacuvajte ih u posebnim promenljivima. Na svakom od skupova. trenirajte jedan model 
i uporedite rezultate po razlicitim skupovima. 
	- Odradio
11. Odgovorite na sledeca pitanja:
	- Na kom podskupu dovijate najbolje performanse predikcije? 
	- Kako se razlikuju predikcije na kompletnom test setu i na parcijalnim?
	- Koji atributi imaju najvecu prediktivnu moc?
	- Kako se razlikuju performanse modela sa optimizovanim parametrima u odnosu na modele sa default parametrima?
	- Da li mislite da bi neka druga kombinacija bila bolja za vas dataset i zasto?
	- Da li su vasi modeli pretrenirani (overfit)?
12. Bonus: 
	- Pronadjite podskup atributa koji maksimizuje performanse prediktivnih algoritama