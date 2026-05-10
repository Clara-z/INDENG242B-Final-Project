# config_companies.py
# Safe GDELT query config.
# Designed to avoid GDELT errors like:
# "One or more of your keywords were too short, too long or too common"
#
# Strategy:
# - Use exact company-name phrases only.
# - Avoid broad/common standalone terms such as "Apple", "Meta", "Amazon".
# - Avoid nested Boolean query terms such as '"Apple" AND (...)'.
# - Keep entity_match_patterns broader than query_terms so downloaded text can still be validated.

COMPANIES = [
    {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "query_terms": ['"Apple Inc"', '"Apple earnings"', '"Apple stock"'],
        "exclude_terms": ['"apple pie"', '"apple orchard"', '"apple cider"', '"Apple Records"'],
        "entity_match_patterns": ["Apple Inc", "Apple", "iPhone", "MacBook", "Apple Watch", "Tim Cook"],
    },
    {
        "ticker": "AAL",
        "company_name": "American Airlines Group Inc.",
        "query_terms": ['"American Airlines Group"', '"American Airlines earnings"', '"American Airlines stock"'],
        "exclude_terms": [],
        "entity_match_patterns": ["American Airlines Group", "American Airlines", "AAL"],
    },
    {
        "ticker": "ABEO",
        "company_name": "Abeona Therapeutics Inc.",
        "query_terms": ['"Abeona Therapeutics"', '"Abeona Therapeutics stock"'],
        "exclude_terms": [],
        "entity_match_patterns": ["Abeona Therapeutics", "Abeona", "ABEO"],
    },
    {
        "ticker": "ABNB",
        "company_name": "Airbnb Inc.",
        "query_terms": ['"Airbnb Inc"', '"Airbnb earnings"', '"Airbnb stock"'],
        "exclude_terms": [],
        "entity_match_patterns": ["Airbnb Inc", "Airbnb", "ABNB", "Brian Chesky"],
    },
    {
        "ticker": "ACAD",
        "company_name": "ACADIA Pharmaceuticals Inc.",
        "query_terms": ['"ACADIA Pharmaceuticals"', '"ACADIA Pharmaceuticals stock"'],
        "exclude_terms": ['"Acadia National Park"', '"Acadia University"'],
        "entity_match_patterns": ["ACADIA Pharmaceuticals", "Acadia Pharmaceuticals", "ACAD", "Nuplazid"],
    },
    {
        "ticker": "ACGL",
        "company_name": "Arch Capital Group Ltd.",
        "query_terms": ['"Arch Capital Group"', '"Arch Capital Group earnings"', '"Arch Capital Group stock"'],
        "exclude_terms": [],
        "entity_match_patterns": ["Arch Capital Group", "Arch Capital", "ACGL"],
    },
    {
        "ticker": "ACHC",
        "company_name": "Acadia Healthcare Company Inc.",
        "query_terms": ['"Acadia Healthcare Company"', '"Acadia Healthcare earnings"', '"Acadia Healthcare stock"'],
        "exclude_terms": ['"Acadia National Park"', '"Acadia University"'],
        "entity_match_patterns": ["Acadia Healthcare Company", "Acadia Healthcare", "ACHC"],
    },
    {
        "ticker": "ACHV",
        "company_name": "Achieve Life Sciences Inc.",
        "query_terms": ['"Achieve Life Sciences"', '"Achieve Life Sciences stock"', '"cytisinicline"'],
        "exclude_terms": [],
        "entity_match_patterns": ["Achieve Life Sciences", "ACHV", "cytisinicline"],
    },
    {
        "ticker": "ACLS",
        "company_name": "Axcelis Technologies Inc.",
        "query_terms": ['"Axcelis Technologies"', '"Axcelis Technologies earnings"', '"Axcelis Technologies stock"'],
        "exclude_terms": [],
        "entity_match_patterns": ["Axcelis Technologies", "Axcelis", "ACLS"],
    },
    {
        "ticker": "ACMR",
        "company_name": "ACM Research Inc.",
        "query_terms": ['"ACM Research Inc"', '"ACM Research earnings"', '"ACM Research stock"'],
        "exclude_terms": ['"Association for Computing Machinery"'],
        "entity_match_patterns": ["ACM Research Inc", "ACM Research", "ACMR"],
    },
    {
        "ticker": "ACRS",
        "company_name": "Aclaris Therapeutics Inc.",
        "query_terms": ['"Aclaris Therapeutics"', '"Aclaris Therapeutics stock"'],
        "exclude_terms": [],
        "entity_match_patterns": ["Aclaris Therapeutics", "Aclaris", "ACRS"],
    },
    {
        "ticker": "ADBE",
        "company_name": "Adobe Inc.",
        "query_terms": ['"Adobe Inc"', '"Adobe earnings"', '"Adobe stock"', '"Adobe Systems"'],
        "exclude_terms": ['"adobe brick"', '"adobe house"'],
        "entity_match_patterns": ["Adobe Inc", "Adobe", "Adobe Systems", "Creative Cloud", "Adobe Firefly", "Shantanu Narayen"],
    },
    {
        "ticker": "ADI",
        "company_name": "Analog Devices Inc.",
        "query_terms": ['"Analog Devices Inc"', '"Analog Devices earnings"', '"Analog Devices stock"'],
        "exclude_terms": [],
        "entity_match_patterns": ["Analog Devices Inc", "Analog Devices", "ADI"],
    },
    {
        "ticker": "ADIL",
        "company_name": "Adial Pharmaceuticals Inc.",
        "query_terms": ['"Adial Pharmaceuticals"', '"Adial Pharmaceuticals stock"'],
        "exclude_terms": [],
        "entity_match_patterns": ["Adial Pharmaceuticals", "Adial", "ADIL"],
    },
    {
        "ticker": "ADMA",
        "company_name": "ADMA Biologics Inc.",
        "query_terms": ['"ADMA Biologics"', '"ADMA Biologics earnings"', '"ADMA Biologics stock"'],
        "exclude_terms": [],
        "entity_match_patterns": ["ADMA Biologics", "ADMA"],
    },
    {
        "ticker": "ADP",
        "company_name": "Automatic Data Processing Inc.",
        "query_terms": ['"Automatic Data Processing"', '"Automatic Data Processing earnings"', '"Automatic Data Processing stock"'],
        "exclude_terms": ['"ADP employment report"', '"ADP jobs report"'],
        "entity_match_patterns": ["Automatic Data Processing", "ADP"],
    },
    {
        "ticker": "ADSK",
        "company_name": "Autodesk Inc.",
        "query_terms": ['"Autodesk Inc"', '"Autodesk earnings"', '"Autodesk stock"', '"AutoCAD"'],
        "exclude_terms": [],
        "entity_match_patterns": ["Autodesk Inc", "Autodesk", "ADSK", "AutoCAD"],
    },
    {
        "ticker": "AEHR",
        "company_name": "Aehr Test Systems",
        "query_terms": ['"Aehr Test Systems"', '"Aehr Test Systems earnings"', '"Aehr Test Systems stock"'],
        "exclude_terms": [],
        "entity_match_patterns": ["Aehr Test Systems", "Aehr", "AEHR"],
    },
    {
        "ticker": "AEIS",
        "company_name": "Advanced Energy Industries Inc.",
        "query_terms": ['"Advanced Energy Industries"', '"Advanced Energy Industries earnings"', '"Advanced Energy Industries stock"'],
        "exclude_terms": [],
        "entity_match_patterns": ["Advanced Energy Industries", "AEIS"],
    },
    {
        "ticker": "AEP",
        "company_name": "American Electric Power Company Inc.",
        "query_terms": ['"American Electric Power"', '"American Electric Power earnings"', '"American Electric Power stock"'],
        "exclude_terms": [],
        "entity_match_patterns": ["American Electric Power Company", "American Electric Power", "AEP"],
    },
    {
        "ticker": "AERO",
        "company_name": "AERO-listed company",
        "query_terms": ['"AERO stock"', '"AERO earnings"'],
        "exclude_terms": ['"aero club"', '"aero show"'],
        "entity_match_patterns": ["AERO"],
    },
    {
        "ticker": "CCO",
        "company_name": "Clear Channel Outdoor Holdings Inc.",
        "query_terms": ['"Clear Channel Outdoor Holdings"', '"Clear Channel Outdoor earnings"', '"Clear Channel Outdoor stock"'],
        "exclude_terms": [],
        "entity_match_patterns": ["Clear Channel Outdoor Holdings", "Clear Channel Outdoor", "CCO"],
    },
    {
        "ticker": "CCRN",
        "company_name": "Cross Country Healthcare Inc.",
        "query_terms": ['"Cross Country Healthcare"', '"Cross Country Healthcare earnings"', '"Cross Country Healthcare stock"'],
        "exclude_terms": ['"cross country race"', '"cross-country race"'],
        "entity_match_patterns": ["Cross Country Healthcare", "CCRN"],
    },
    {
        "ticker": "CCS",
        "company_name": "Century Communities Inc.",
        "query_terms": ['"Century Communities Inc"', '"Century Communities earnings"', '"Century Communities stock"'],
        "exclude_terms": ['"carbon capture"', '"CCS technology"'],
        "entity_match_patterns": ["Century Communities Inc", "Century Communities", "CCS"],
    },
    {
        "ticker": "TSLA",
        "company_name": "Tesla Inc.",
        "query_terms": ['"Tesla Inc"', '"Tesla earnings"', '"Tesla stock"', '"TSLA stock"'],
        "exclude_terms": ['"Nikola Tesla"', '"Tesla coil"'],
        "entity_match_patterns": ["Tesla Inc", "Tesla", "TSLA", "Model 3", "Model Y", "Elon Musk"],
    },
    {
        "ticker": "NFLX",
        "company_name": "Netflix Inc.",
        "query_terms": ['"Netflix Inc"', '"Netflix earnings"', '"Netflix stock"', '"NFLX stock"'],
        "exclude_terms": [],
        "entity_match_patterns": ["Netflix Inc", "Netflix", "NFLX", "Reed Hastings", "Ted Sarandos"],
    },
    {
        "ticker": "MSFT",
        "company_name": "Microsoft Corporation",
        "query_terms": ['"Microsoft Corporation"', '"Microsoft earnings"', '"Microsoft stock"', '"MSFT stock"'],
        "exclude_terms": [],
        "entity_match_patterns": ["Microsoft Corporation", "Microsoft", "MSFT", "Azure", "Satya Nadella"],
    },
    {
        "ticker": "META",
        "company_name": "Meta Platforms Inc.",
        "query_terms": ['"Meta Platforms"', '"Meta Platforms earnings"', '"Meta Platforms stock"', '"META stock"'],
        "exclude_terms": ['"meta analysis"', '"metadata"', '"metaphysics"'],
        "entity_match_patterns": ["Meta Platforms", "Meta", "Facebook", "Instagram", "Mark Zuckerberg", "META"],
    },
    {
        "ticker": "GOOGL",
        "company_name": "Alphabet Inc.",
        "query_terms": ['"Alphabet Inc"', '"Alphabet earnings"', '"Google earnings"', '"GOOGL stock"'],
        "exclude_terms": ['"alphabet soup"'],
        "entity_match_patterns": ["Alphabet Inc", "Alphabet", "Google", "GOOGL", "Sundar Pichai"],
    },
    {
        "ticker": "AMZN",
        "company_name": "Amazon.com Inc.",
        "query_terms": ['"Amazon.com Inc"', '"Amazon earnings"', '"Amazon stock"', '"AMZN stock"'],
        "exclude_terms": ['"Amazon rainforest"', '"Amazon river"'],
        "entity_match_patterns": ["Amazon.com Inc", "Amazon.com", "Amazon", "Amazon Web Services", "AWS", "AMZN", "Andy Jassy", "Jeff Bezos"],
    },
]

SELECTED_TICKERS = [c["ticker"] for c in COMPANIES]
