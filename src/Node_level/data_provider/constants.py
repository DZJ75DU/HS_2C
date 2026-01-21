import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))
data_root = f'{project_root}/0_baseline/generated_data'

DEFAULT_DATASET_PATHS = {
    'ogb_arxiv': f'{data_root}/20250606_ogb_arxiv',
    'ogb_arxiv_hete': f'{data_root}/20250606_ogb_arxiv_hete',
    'citeseer': f'{data_root}/20250606_citeseer',
    'reddit': f'{data_root}/20250606_reddit',
    'instagram': f'{data_root}/20250606_instagram',
    'arxiv_2023': f'{data_root}/20250606_arxiv_2023',
    'products': f'{data_root}/20250606_products',
    'book_hete': f'{data_root}/20250606_book_hete',
    'book_homo': f'{data_root}/20250606_book_homo',
    'dblp_hete': f'{data_root}/20250606_dblp_hete',
    'dblp_homo': f'{data_root}/20250606_dblp_homo'
}

CATEGORY_LISTS = {
    'arxiv': """ Artificial Intelligence; Hardware Architecture; Computational Complexity; Computational Engineering, Finance, and Science; Computational Geometry; Computation and Language; Cryptography and Security; Computer Vision and Pattern Recognition; Computers and Society; Databases; Distributed, Parallel, and Cluster Computing; Digital Libraries; Discrete Mathematics; Data Structures and Algorithms; Emerging Technologies; Formal Languages and Automata Theory; General Literature; Graphics; Computer Science and Game Theory; Human-Computer Interaction; Information Retrieval; Information Theory; Machine Learning; Logic in Computer Science; Multiagent Systems; Multimedia; Mathematical Software; Numerical Analysis; Neural and Evolutionary Computing; Networking and Internet Architecture; Other Computer Science; Operating Systems; Performance; Programming Languages; Robotics; Symbolic Computation; Sound; Software Engineering; Social and Information Networks; Systems and Control """,
    'dblp': """ Artificial Intelligence and Pattern Recognition; Computer Networks; Network and Information Security; Database and Data Mining; System Software and Software Engineering; Computer Graphics and Multimedia; Human-Computer Interaction and Pervasive Computing; Theoretical Computer Science; High-Performance Computing """,
    'instagram': """ Normal Users; Commercial Users """,
    'citeseer': """ Information Retrieval; Human-Computer Interaction; Agents; Artificial Intelligence; Machine Learning; Databases """,
    'reddit': """ Normal Users; Popular Users """,
    'products': """ Home & Kitchen; Health & Personal Care; Beauty; Sports & Outdoors; Books; Patio, Lawn & Garden; Toys & Games; CDs & Vinyl; Cell Phones & Accessories; Grocery & Gourmet Food; Arts, Crafts & Sewing; Clothing, Shoes & Jewelry; Electronics; Movies & TV; Software; Video Games; Automotive; Pet Supplies; Office Products; Industrial & Scientific; Musical Instruments; Tools & Home Improvement; Magazine Subscriptions; Baby Products; Appliances; Kitchen & Dining; Collectibles & Fine Art; All Beauty; Luxury Beauty; Amazon Fashion; Computers; All Electronics; Purchase Circles; MP3 Players & Accessories; Gift Cards; Office & School Supplies; Home Improvement; Camera & Photo; GPS & Navigation; Digital Music; Car Electronics; Baby; Kindle Store; Buy a Kindle; Furniture & decoration """,
    'book': """ children; comics, graphic; fantasy, paranormal; history, historical fiction, biography; mystery, thriller, crime; poetry; romance; young-adult """,
}


LABEL_WHOL_DICT = {
    'arxiv': {
        0: "Numerical Analysis",
        1: "Multimedia",
        2: "Logic in Computer Science",
        3: "Computers and Society",
        4: "Cryptography and Security",
        5: "Distributed, Parallel, and Cluster Computing",
        6: "Human-Computer Interaction",
        7: "Computational Engineering, Finance, and Science",
        8: "Networking and Internet Architecture",
        9: "Computational Complexity",
        10: "Artificial Intelligence",
        11: "Multiagent Systems",
        12: "General Literature",
        13: "Neural and Evolutionary Computing",
        14: "Symbolic Computation",
        15: "Hardware Architecture",
        16: "Computer Vision and Pattern Recognition",
        17: "Graphics",
        18: "Emerging Technologies",
        19: "Systems and Control",
        20: "Computational Geometry",
        21: "Other Computer Science",
        22: "Programming Languages",
        23: "Software Engineering",
        24: "Machine Learning",
        25: "Sound",
        26: "Social and Information Networks",
        27: "Robotics",
        28: "Information Theory",
        29: "Performance",
        30: "Computation and Language",
        31: "Information Retrieval",
        32: "Mathematical Software",
        33: "Formal Languages and Automata Theory",
        34: "Data Structures and Algorithms",
        35: "Operating Systems",
        36: "Computer Science and Game Theory",
        37: "Databases",
        38: "Digital Libraries",
        39: "Discrete Mathematics",
    },
    'dblp': {
        0: "Artificial Intelligence and Pattern Recognition",
        1: "Computer Networks",
        2: "Network and Information Security",
        3: "Database and Data Mining",
        4: "System Software and Software Engineering",
        5: "Computer Graphics and Multimedia",
        6: "Human-Computer Interaction and Pervasive Computing",
        7: "Theoretical Computer Science",
        8: "High-Performance Computing"
    },
    'products': {
        0: "Home & Kitchen",
        1: "Health & Personal Care",
        2: "Beauty",
        3: "Sports & Outdoors",
        4: "Books",
        5: "Patio, Lawn & Garden",
        6: "Toys & Games",
        7: "CDs & Vinyl",
        8: "Cell Phones & Accessories",
        9: "Grocery & Gourmet Food",
        10: "Arts, Crafts & Sewing",
        11: "Clothing, Shoes & Jewelry",
        12: "Electronics",
        13: "Movies & TV",
        14: "Software",
        15: "Video Games",
        16: "Automotive",
        17: "Pet Supplies",
        18: "Office Products",
        19: "Industrial & Scientific",
        20: "Musical Instruments",
        21: "Tools & Home Improvement",
        23: "Baby Products",
        24: "label 25",
        25: "Appliances",
        26: "Kitchen & Dining",
        27: "Collectibles & Fine Art",
        28: "All Beauty",
        30: "Amazon Fashion",
        31: "Computers",
        32: "All Electronics",
        34: "MP3 Players & Accessories",
        36: "Office & School Supplies",
        37: "Home Improvement",
        38: "Camera & Photo",
        39: "GPS & Navigation",
        40: "Digital Music",
        41: "Car Electronics",
        42: "Baby",
    },
    'instagram': {
        0: "Normal Users",
        1: "Commercial Users",
    },
    'citeseer': {
        0: "Agents",
        1: "ML (Machine Learning)",
        2: "IR (Information Retrieval)",
        3: "DB (Databases)",
        4: "HCI (Human-Computer Interaction)",
        5: "AI (Artificial Intelligence)",
    },
    'reddit': {
        0: "Normal Users",
        1: "Popular Users",
    },
    'book': {
        0: "children",
        1: "comics, graphic",
        2: "fantasy, paranormal",
        3: "history, historical fiction, biography",
        4: "mystery, thriller, crime",
        5: "poetry",
        6: "romance",
        7: "young-adult"
    },
}


LABEL_ABBR_DICT = {
    'arxiv': {
        0: "cs.NA",
        1: "cs.MM",
        2: "cs.LO",
        3: "cs.CY",
        4: "cs.CR",
        5: "cs.DC",
        6: "cs.HC",
        7: "cs.CE",
        8: "cs.NI",
        9: "cs.CC",
        10: "cs.AI",
        11: "cs.MA",
        12: "cs.GL",
        13: "cs.NE",
        14: "cs.SC",
        15: "cs.AR",
        16: "cs.CV",
        17: "cs.GR",
        18: "cs.ET",
        19: "cs.SY",
        20: "cs.CG",
        21: "cs.OH",
        22: "cs.PL",
        23: "cs.SE",
        24: "cs.LG",
        25: "cs.SD",
        26: "cs.SI",
        27: "cs.RO",
        28: "cs.IT",
        29: "cs.PF",
        30: "cs.CL",
        31: "cs.IR",
        32: "cs.MS",
        33: "cs.FL",
        34: "cs.DS",
        35: "cs.OS",
        36: "cs.GT",
        37: "cs.DB",
        38: "cs.DL",
        39: "cs.DM",
    },
    'dblp': {},
    'products': {},
    'cora': {},
    'pubmed': {},
    'instagram': {},
    'citeseer': {},
    'reddit': {},
    'book': {},
}

# Tokens
BACKGROUND_INFO_DEFAULT_TOKEN = '<BACKGROUND_NODE_INFO>'
TARGET_INFO_DEFAULT_TOKEN = '<TARGET_NODE_INFO>'
DEFAULT_P_TITLE_TOKEN = "<paper_title>"
DEFAULT_P_ABSTRACT_TOKEN = "<paper_abstract>"
DEFAULT_CATEGORY_TOKEN = "<category_list>"
DEFAULT_G_NEIGHBOR_INFO_TOKEN = "<NEIGHBOR_INFO>"
DEFAULT_CONDENSATION_FAILURE_TOKEN = "<CONDENSATION_FAILURE>"
DEFAULT_SUMMARY_CORPUS_TOKEN = "<SUMMARY_CORPUS>"
DEFAULT_DATASET_NAME_TOKEN = "<DATASET_TOKEN>"
DEFAULT_GRAPH_TYPE_TOKEN = "<GRAPH_TYPE_TOKEN>"
DEFAULT_NODE_TYPE_TOKEN = "<NODE_TYPE_TOKEN>"
DEFAULT_CENTER_RAW_TEXT_TOKEN = "<CENTER_RAW_TEXT>"
DEFAULT_CORPUS_RAW_TEXT_TOKEN = "<SUMMARY_CORPUS>"
DEFAULT_SUMMARY_RAW_TEXT_TOKEN = "<SUMMARY_RAW_TEXT>"
