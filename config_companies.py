# config_companies.py
# Selected tickers from discover_gdelt_consistent_tickers.py
# Used by full-text GDELT scraping pipeline.

COMPANIES = [
    {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "query_terms": [
            '"Apple Inc"', '"Apple" AND (stock OR shares OR earnings OR revenue)',
            '"Apple iPhone"', '"Apple MacBook"', '"Apple Watch"', '"Tim Cook"'
        ],
        "exclude_terms": ['"apple pie"', '"apple orchard"', '"apple cider"', '"Apple Records"'],
        "entity_match_patterns": ["Apple", "Apple Inc", "iPhone", "MacBook", "Apple Watch", "Tim Cook"],
    },
    {
        "ticker": "AAL",
        "company_name": "American Airlines Group Inc.",
        "query_terms": [
            '"American Airlines"', '"American Airlines Group"',
            '"AAL stock"', '"American Airlines earnings"', '"Robert Isom"'
        ],
        "exclude_terms": [],
        "entity_match_patterns": ["American Airlines", "American Airlines Group", "AAL", "Robert Isom"],
    },
    {
        "ticker": "ABEO",
        "company_name": "Abeona Therapeutics Inc.",
        "query_terms": [
            '"Abeona Therapeutics"', '"Abeona" AND (stock OR shares OR earnings OR trial OR FDA)',
            '"ABEO stock"'
        ],
        "exclude_terms": [],
        "entity_match_patterns": ["Abeona Therapeutics", "Abeona", "ABEO"],
    },
    {
        "ticker": "ABNB",
        "company_name": "Airbnb Inc.",
        "query_terms": [
            '"Airbnb"', '"Airbnb Inc"', '"ABNB stock"',
            '"Airbnb earnings"', '"Brian Chesky"'
        ],
        "exclude_terms": [],
        "entity_match_patterns": ["Airbnb", "Airbnb Inc", "ABNB", "Brian Chesky"],
    },
    {
        "ticker": "ACAD",
        "company_name": "ACADIA Pharmaceuticals Inc.",
        "query_terms": [
            '"ACADIA Pharmaceuticals"', '"Acadia Pharmaceuticals"',
            '"ACAD stock"', '"Nuplazid"'
        ],
        "exclude_terms": ['"Acadia National Park"', '"Acadia University"'],
        "entity_match_patterns": ["ACADIA Pharmaceuticals", "Acadia Pharmaceuticals", "ACAD", "Nuplazid"],
    },
    {
        "ticker": "ACGL",
        "company_name": "Arch Capital Group Ltd.",
        "query_terms": [
            '"Arch Capital Group"', '"Arch Capital" AND (stock OR shares OR earnings OR insurance)',
            '"ACGL stock"'
        ],
        "exclude_terms": [],
        "entity_match_patterns": ["Arch Capital Group", "Arch Capital", "ACGL"],
    },
    {
        "ticker": "ACHC",
        "company_name": "Acadia Healthcare Company Inc.",
        "query_terms": [
            '"Acadia Healthcare"', '"Acadia Healthcare Company"',
            '"ACHC stock"', '"Acadia Healthcare earnings"'
        ],
        "exclude_terms": ['"Acadia National Park"', '"Acadia University"'],
        "entity_match_patterns": ["Acadia Healthcare", "Acadia Healthcare Company", "ACHC"],
    },
    {
        "ticker": "ACHV",
        "company_name": "Achieve Life Sciences Inc.",
        "query_terms": [
            '"Achieve Life Sciences"', '"Achieve Life Sciences" AND (stock OR shares OR trial OR FDA)',
            '"ACHV stock"', '"cytisinicline"'
        ],
        "exclude_terms": [],
        "entity_match_patterns": ["Achieve Life Sciences", "ACHV", "cytisinicline"],
    },
    {
        "ticker": "ACLS",
        "company_name": "Axcelis Technologies Inc.",
        "query_terms": [
            '"Axcelis Technologies"', '"Axcelis" AND (stock OR shares OR earnings OR semiconductor)',
            '"ACLS stock"'
        ],
        "exclude_terms": [],
        "entity_match_patterns": ["Axcelis Technologies", "Axcelis", "ACLS"],
    },
    {
        "ticker": "ACMR",
        "company_name": "ACM Research Inc.",
        "query_terms": [
            '"ACM Research"', '"ACM Research Inc"',
            '"ACMR stock"', '"ACM Research earnings"'
        ],
        "exclude_terms": ['"Association for Computing Machinery"'],
        "entity_match_patterns": ["ACM Research", "ACMR"],
    },
    {
        "ticker": "ACRS",
        "company_name": "Aclaris Therapeutics Inc.",
        "query_terms": [
            '"Aclaris Therapeutics"', '"Aclaris" AND (stock OR shares OR earnings OR trial OR FDA)',
            '"ACRS stock"'
        ],
        "exclude_terms": [],
        "entity_match_patterns": ["Aclaris Therapeutics", "Aclaris", "ACRS"],
    },
    {
        "ticker": "ADBE",
        "company_name": "Adobe Inc.",
        "query_terms": [
            '"Adobe" AND (stock OR shares OR earnings OR revenue)',
            '"Adobe Inc"', '"Adobe Systems"', '"Adobe Creative Cloud"',
            '"Adobe Firefly"', '"Shantanu Narayen"'
        ],
        "exclude_terms": ['"adobe brick"', '"adobe house"'],
        "entity_match_patterns": ["Adobe", "Adobe Inc", "Adobe Systems", "Creative Cloud", "Adobe Firefly", "Shantanu Narayen"],
    },
    {
        "ticker": "ADI",
        "company_name": "Analog Devices Inc.",
        "query_terms": [
            '"Analog Devices"', '"Analog Devices Inc"',
            '"ADI stock"', '"Analog Devices earnings"'
        ],
        "exclude_terms": [],
        "entity_match_patterns": ["Analog Devices", "Analog Devices Inc", "ADI"],
    },
    {
        "ticker": "ADIL",
        "company_name": "Adial Pharmaceuticals Inc.",
        "query_terms": [
            '"Adial Pharmaceuticals"', '"Adial" AND (stock OR shares OR clinical OR trial OR FDA)',
            '"ADIL stock"'
        ],
        "exclude_terms": [],
        "entity_match_patterns": ["Adial Pharmaceuticals", "Adial", "ADIL"],
    },
    {
        "ticker": "ADMA",
        "company_name": "ADMA Biologics Inc.",
        "query_terms": [
            '"ADMA Biologics"', '"ADMA" AND (Biologics OR stock OR shares OR earnings)',
            '"ADMA stock"'
        ],
        "exclude_terms": [],
        "entity_match_patterns": ["ADMA Biologics", "ADMA"],
    },
    {
        "ticker": "ADP",
        "company_name": "Automatic Data Processing Inc.",
        "query_terms": [
            '"Automatic Data Processing"', '"ADP" AND (stock OR shares OR earnings OR payroll)',
            '"ADP earnings"'
        ],
        "exclude_terms": ['"ADP report"', '"ADP employment report"'],
        "entity_match_patterns": ["Automatic Data Processing", "ADP", "ADP payroll"],
    },
    {
        "ticker": "ADSK",
        "company_name": "Autodesk Inc.",
        "query_terms": [
            '"Autodesk"', '"Autodesk Inc"', '"ADSK stock"',
            '"Autodesk earnings"', '"AutoCAD"'
        ],
        "exclude_terms": [],
        "entity_match_patterns": ["Autodesk", "Autodesk Inc", "ADSK", "AutoCAD"],
    },
    {
        "ticker": "AEHR",
        "company_name": "Aehr Test Systems",
        "query_terms": [
            '"Aehr Test Systems"', '"Aehr" AND (stock OR shares OR earnings OR semiconductor)',
            '"AEHR stock"'
        ],
        "exclude_terms": [],
        "entity_match_patterns": ["Aehr Test Systems", "Aehr", "AEHR"],
    },
    {
        "ticker": "AEIS",
        "company_name": "Advanced Energy Industries Inc.",
        "query_terms": [
            '"Advanced Energy Industries"', '"Advanced Energy" AND (AEIS OR stock OR shares OR earnings)',
            '"AEIS stock"'
        ],
        "exclude_terms": ['"advanced energy" "renewable"', '"advanced energy" "policy"'],
        "entity_match_patterns": ["Advanced Energy Industries", "AEIS"],
    },
    {
        "ticker": "AEP",
        "company_name": "American Electric Power Company Inc.",
        "query_terms": [
            '"American Electric Power"', '"American Electric Power Company"',
            '"AEP stock"', '"AEP earnings"'
        ],
        "exclude_terms": [],
        "entity_match_patterns": ["American Electric Power", "American Electric Power Company", "AEP"],
    },
    {
        "ticker": "AERO",
        "company_name": "AeroVironment or AERO-listed company",
        "query_terms": [
            '"AERO stock"', '"AERO" AND (stock OR shares OR earnings)',
            '"Aero" AND (Nasdaq OR NYSE OR earnings OR shares)'
        ],
        "exclude_terms": ['"aero club"', '"aero show"', '"aerospace industry"'],
        "entity_match_patterns": ["AERO"],
    },
    {
        "ticker": "CCO",
        "company_name": "Clear Channel Outdoor Holdings Inc.",
        "query_terms": [
            '"Clear Channel Outdoor"', '"Clear Channel Outdoor Holdings"',
            '"CCO stock"', '"Clear Channel Outdoor earnings"'
        ],
        "exclude_terms": [],
        "entity_match_patterns": ["Clear Channel Outdoor", "Clear Channel Outdoor Holdings", "CCO"],
    },
    {
        "ticker": "CCRN",
        "company_name": "Cross Country Healthcare Inc.",
        "query_terms": [
            '"Cross Country Healthcare"', '"Cross Country Healthcare Inc"',
            '"CCRN stock"', '"Cross Country Healthcare earnings"'
        ],
        "exclude_terms": ['"cross country race"', '"cross-country race"'],
        "entity_match_patterns": ["Cross Country Healthcare", "CCRN"],
    },
    {
        "ticker": "CCS",
        "company_name": "Century Communities Inc.",
        "query_terms": [
            '"Century Communities"', '"Century Communities Inc"',
            '"CCS stock"', '"Century Communities earnings"'
        ],
        "exclude_terms": ['"carbon capture"', '"CCS technology"'],
        "entity_match_patterns": ["Century Communities", "Century Communities Inc", "CCS"],
    },
    {
        "ticker": "TSLA",
        "company_name": "Tesla Inc.",
        "query_terms": [
            '"Tesla" AND (stock OR shares OR earnings OR revenue OR deliveries)',
            '"Tesla Inc"', '"Tesla Model 3"', '"Tesla Model Y"',
            '"Elon Musk" AND Tesla', '"TSLA stock"'
        ],
        "exclude_terms": ['"Nikola Tesla"', '"Tesla coil"'],
        "entity_match_patterns": ["Tesla", "Tesla Inc", "TSLA", "Model 3", "Model Y", "Elon Musk"],
    },
    {
        "ticker": "NFLX",
        "company_name": "Netflix Inc.",
        "query_terms": [
            '"Netflix"', '"Netflix Inc"', '"NFLX stock"',
            '"Netflix earnings"', '"Reed Hastings"', '"Ted Sarandos"'
        ],
        "exclude_terms": [],
        "entity_match_patterns": ["Netflix", "Netflix Inc", "NFLX", "Reed Hastings", "Ted Sarandos"],
    },
    {
        "ticker": "MSFT",
        "company_name": "Microsoft Corporation",
        "query_terms": [
            '"Microsoft"', '"Microsoft Corporation"', '"MSFT stock"',
            '"Microsoft earnings"', '"Azure"', '"Satya Nadella"'
        ],
        "exclude_terms": [],
        "entity_match_patterns": ["Microsoft", "Microsoft Corporation", "MSFT", "Azure", "Satya Nadella"],
    },
    {
        "ticker": "META",
        "company_name": "Meta Platforms Inc.",
        "query_terms": [
            '"Meta Platforms"', '"Meta" AND (stock OR shares OR earnings OR revenue)',
            '"Facebook" AND (Meta OR stock OR earnings)', '"Instagram" AND Meta',
            '"Mark Zuckerberg"', '"META stock"'
        ],
        "exclude_terms": ['"meta analysis"', '"metadata"', '"metaphysics"'],
        "entity_match_patterns": ["Meta Platforms", "Meta", "Facebook", "Instagram", "Mark Zuckerberg", "META"],
    },
    {
        "ticker": "GOOGL",
        "company_name": "Alphabet Inc.",
        "query_terms": [
            '"Alphabet Inc"', '"Alphabet" AND (Google OR stock OR shares OR earnings)',
            '"Google" AND (stock OR shares OR earnings OR revenue)',
            '"GOOGL stock"', '"Sundar Pichai"'
        ],
        "exclude_terms": ['"alphabet soup"'],
        "entity_match_patterns": ["Alphabet Inc", "Alphabet", "Google", "GOOGL", "Sundar Pichai"],
    },
    {
        "ticker": "AMZN",
        "company_name": "Amazon.com Inc.",
        "query_terms": [
            '"Amazon.com"', '"Amazon" AND (stock OR shares OR earnings OR revenue)',
            '"Amazon Web Services"', '"AWS" AND Amazon',
            '"AMZN stock"', '"Andy Jassy"', '"Jeff Bezos"'
        ],
        "exclude_terms": ['"Amazon rainforest"', '"Amazon river"'],
        "entity_match_patterns": ["Amazon", "Amazon.com", "Amazon Web Services", "AWS", "AMZN", "Andy Jassy", "Jeff Bezos"],
    },
]

SELECTED_TICKERS = [c["ticker"] for c in COMPANIES]
