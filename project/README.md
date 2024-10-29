Project proposal for AI Development

Returer inom e-handelsindustrin för kläder i Europa är en betydande utmaning för många återförsäljare. I genomsnitt ligger returprocenten kring 55%, vilket innebär att mer än hälften av de sålda klädesplaggen returneras av kunderna. Denna höga siffra kan delvis förklaras av faktorer som storleksproblem, felaktiga produktbeskrivningar och kunden som vill prova kläder innan de gör ett slutgiltigt beslut. 

Den höga returprocenten skapar problem för företagens balansräkningar och ekonomiska resultat. Kostnaderna för hantering och frakt av returer kan snabbt äta upp vinsterna, vilket påverkar företagets lönsamhet. Dessutom kan en ökad returandel leda till lägre intäkter och en mer komplex lagerhantering, vilket i sin tur kan påverka kassaflödet negativt. Dessa faktorer gör det nödvändigt för företag att noggrant övervaka sina returprocesser och utveckla strategier för att minska returfrekvensen, för att säkerställa en sund ekonomisk ställning.

Maskinlärning ger företag en möjlighet att kunna förutsäga vilka beställningar som riskerar att bli returnerade, och kan eventuellt identifiera och blockera vissa köpare från att utnyttja den 14-dagars fria ångerrätt som gäller i exempelvis Sverige, och i sin tur kan hjälpa till att minimera kostnader och förbättra lönsamheten för företag.

Tanken med mitt projekt blir därför en två-stegs lösning:
1. Identifiera ordrar som riskerar att bli returnerade, och beräkna ordervärde, och
2. Identifiera potentiella återkommande kunder som rekommenderas att blockeras.

Data kommer jag ha tillgång till från bekantas e-handels webshop med fokus på försäljning av herrkläder (kostymer, skjortor, etc). Datan kommer att anonymiseras för GDPR och av privataiserings skäl. Ca 3000 rader ordrar, 5000 rader produktrader (alltså om en order innehåller 2 olika produkter), och några tusen rader kunddata finns att tillgå. Order datan är labeled för vilka ordrar som har returnerats historiskt. Features som ingår i datan inkluderar storlekar, modell, typ av produkt, antal, betalmetod, fraktkostnad, adress, pris, rabatter, etc. Vanlig e-handels data helt enkelt, baserad på verklig e-handel. Totalt finns 75 features att tillgå, där de mest relevanta väljs ut (exempelvis kommer gatuadress för leverans räknas bort) och ytterligare beräknade dimensioner läggs till (te.x. om en order innehåller samma skjorta i två olika storlekar).

Tanken är att först träna en klassificeringsmodell för punkt 1 och 2 på datasetet, där en form av ensemble model kommer byggas för att avgöra den optimala klassificeringsmodellen, utvärderat efter högst accuracy. Utifrån datamängden vill jag prova logistic regression, kNN, classification tree. Om dessa modeller visar sig ha för dåligt resultat kommer jag gå vidare med någon form av dimension reduction som PCA, alternativt även Neural Networks. Därefter ska ett enkelt gränssnitt byggas för webshops ägarna att kunna ladda upp ny försäljningsdata och få ut klassificering om kund och prediktion om ordrarreturer. Programmet ska kunna visa accuracy data för bästa modellen, samt uppskattat värde av returer (baserat på klassificeringen om retur/ej retur). 

I en ideal värld skulle programmet integreras direkt med ägarnas webshop, men tyvärr är mina tekniska egenskaper begränsad till att köra prediktionen lokalt och manuellt.

