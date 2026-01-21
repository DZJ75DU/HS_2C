GRAPH_DESC = {
    "graph-sst2": {
        "dataset_desc": (
            "Graph-SST2 transforms the Stanford Sentiment Treebank (SST-2) into "
            "graph-structured sentiment classification data. Each sentence is "
            "converted into a dependency tree where nodes represent words and edges "
            "represent syntactic relations. The task is to predict a binary sentiment "
            "label (positive / negative) for each graph."
        ),
        "task_type": "binary_classification",
        "label_dict": {
            0: "negative",
            1: "positive"
        },
        "label_desc": {
            0: "negative: the sentence expresses negative sentiment.",
            1: "positive: the sentence expresses positive sentiment."
        }
    },

    "graph-sst5": {
        "dataset_desc": (
            "Graph-SST5 converts the Stanford Sentiment Treebank (SST-5) into "
            "dependency-tree graph data. Each graph corresponds to a sentence, and "
            "its nodes and edges follow syntactic dependency structures. The task "
            "is to classify fine-grained sentiment with five categories."
        ),
        "task_type": "multi_label_classification",
        "label_dict": {
            0: "very negative",
            1: "negative",
            2: "neutral",
            3: "positive",
            4: "very positive"
        },
        "label_desc": {
            0: "very negative: the sentence expresses strong negative sentiment.",
            1: "negative: the sentence expresses negative sentiment.",
            2: "neutral: the sentence expresses neutral or mixed sentiment.",
            3: "positive: the sentence expresses positive sentiment.",
            4: "very positive: the sentence expresses strong positive sentiment."
        }
    },
    "graph-twitter": {
        "dataset_desc": (
            "Graph-Twitter is a sentiment classification dataset constructed from "
            "Twitter posts, where each tweet is transformed into a dependency-tree "
            "graph. The graph structure captures linguistic and syntactic relations "
            "between tokens. The task is to classify tweets into positive, neutral, or negative "
            "sentiment categories."
        ),
        "task_type": "multi_label_classification",
        "label_dict": {
            0: "negative",
            1: "neutral",
            2: "positive"
        },
        "label_desc": {
            0: "negative: the tweet expresses negative sentiment.",
            1: "neutral: the tweet expresses neutral sentiment.",
            2: "positive: the tweet expresses positive sentiment."
        }
    },
}

MOLECULENET_LABELS = {
    # -----------------------------
    #  Binary classification tasks
    # -----------------------------
    "bbbp": {
        0: "non-permeable",
        1: "permeable",
    },
    "clintox": {
        0: "non-toxic",
        1: "toxic",
    },
    # -----------------------------
    #  Multi-label / Multi-task datasets
    # -----------------------------
    # MUV (17 tasks)
    "muv": [f"MUV_task_{i}" for i in range(17)],
    # SIDER (27 tasks)
    # SIDER 的任务名字是真实的“副作用分类”
    # 这里列出官方任务名称（可直接用）
    "sider": [
        "Infections",
        "Neoplasms",
        "Blood disorders",
        "Immune disorders",
        "Metabolism disorders",
        "Mental disorders",
        "Nervous system disorders",
        "Vision disorders",
        "Hearing disorders",
        "Cardiac disorders",
        "Vascular disorders",
        "Respiratory disorders",
        "Digestive system disorders",
        "Skin disorders",
        "Muscular disorders",
        "Renal disorders",
        "Reproductive system disorders",
        "Pregnancy-related disorders",
        "Congenital disorders",
        "General disorders",
        "Trauma/poisoning",
        "Surgical/medical procedures",
        "Social circumstances",
        "Investigations",
        "Endocrine disorders",
        "Benign neoplasms",
        "Malignant neoplasms",
    ],
}

MOLECULENET_DESC = {
    # ============================================================
    # Binary classification tasks
    # ============================================================
    # BBBP: Binary labels of blood-brain barrier penetration(permeability).
    # The Blood-brain barrier penetration (BBBP) dataset comes from a recent study 52 on the modeling and prediction of the barrier permeability. As a membrane separating circulating blood and brain extracellular fluid, the blood-brain barrier blocks most drugs, hormones and neurotransmitters. Thus penetration of the barrier forms a long-standing issue in development of drugs targeting central nervous system. This dataset includes binary labels for over 2000 compounds on their permeability properties. Scaffold splitting is also recommended for this well-defined target.
    "bbbp": {
        "dataset_desc": "Predict whether a molecule penetrates the blood-brain barrier (BBB).",
        "task_type": "binary_classification",
        "label_dict": {
            0: "non-permeable",
            1: "permeable"
        },
        "label_desc": {
            0: "non-permeable: the compound does not penetrate the blood-brain barrier.",
            1: "permeable: the compound penetrates the blood-brain barrier."
        }
    },

    # ClinTox: Qualitative data of drugs approved by the FDA and those that have failed clinical trials for toxicity reasons.
    # The ClinTox dataset, introduced as part of this work, compares drugs approved by the FDA and drugs that have failed clinical trials for toxicity reasons. The dataset includes two classification tasks for 1491 drug compounds with known chemical structures: (1) clinical trial toxicity (or absence of toxicity) and (2) FDA approval status. List of FDA-approved drugs are compiled from the SWEETLEAD database, and list of drugs that failed clinical trials for toxicity reasons are compiled from the Aggregate Analysis of ClinicalTrials.gov (AACT) database.
    # "clintox": {
    #     "dataset_desc": "Predict whether a drug compound causes clinical toxicity.",
    #     "task_type": "binary_classification",
    #     "label_dict": {
    #         0: "non-toxic",
    #         1: "toxic"
    #     },
    #     "label_desc": {
    #         0: "non-toxic: the drug compound does not cause clinical toxicity.",
    #         1: "toxic: the drug compound causes clinical toxicity."
    #     }
    # },
    "clintox": {
        "dataset_desc": "Predict clinical toxicity and FDA approval status for drug compounds.",
        "task_type": "multi_task_binary_classification",
        "task_order": ["clinical_toxicity", "fda_approval"],
        "tasks": {
            "clinical_toxicity": {
                "dataset_desc": "Predict whether a drug compound causes clinical toxicity.",
                "label_dict": {
                    0: "non-toxic",
                    1: "toxic"
                },
                "label_desc": {
                    0: "non-toxic: the drug compound does not cause clinical toxicity.",
                    1: "toxic: the drug compound causes clinical toxicity."
                }
            },
            "fda_approval": {
                "dataset_desc": "Predict whether a drug compound is approved by the FDA.",
                "label_dict": {
                    0: "not-approved",
                    1: "approved"
                },
                "label_desc": {
                    0: "not-approved: the drug compound is not approved by the FDA.",
                    1: "approved: the drug compound is approved by the FDA."
                }
            }
        }
    },

    # ============================================================
    # Multi-task classification
    # ============================================================  
    # ============================================================
    # MUV: 17 virtual screening tasks
    # ============================================================
    # MUV: Subset of PubChem BioAssay by applying a refined nearest neighbor analysis, designed for validation of virtual screening techniques.
    # The Maximum Unbiased Validation (MUV) group is another benchmark dataset selected from PubChem BioAssay by applying a refined nearest neighbor analysis. The MUV dataset contains 17 challenging tasks for around 90 thousand compounds and is specifically designed for validation of virtual screening techniques.
    # 在 info.py 的 MOLECULENET_DESC 字典中，替换整个 'muv' 条目

    'muv': {
        'dataset_desc': 'Predict whether a molecule is active in 17 PubChem BioAssay tasks selected for maximum unbiased validation.',
        'task_type': 'multi_task_binary_classification',
        # Explicit task order matching CSV columns
        'task_order': [
            'MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652',
            'MUV-689', 'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733',
            'MUV-737', 'MUV-810', 'MUV-832', 'MUV-846', 'MUV-852',
            'MUV-858', 'MUV-859'
        ],
        'tasks': {
            'MUV-466': {
                'dataset_desc': 'Predict whether a molecule is active in bioassay MUV-466.',
                'label_dict': {0: 'inactive', 1: 'active'},
                'label_desc': {
                    0: 'inactive: the molecule is inactive in bioassay MUV-466.',
                    1: 'active: the molecule is active in bioassay MUV-466.'
                }
            },
            'MUV-548': {
                'dataset_desc': 'Predict whether a molecule is active in bioassay MUV-548.',
                'label_dict': {0: 'inactive', 1: 'active'},
                'label_desc': {
                    0: 'inactive: the molecule is inactive in bioassay MUV-548.',
                    1: 'active: the molecule is active in bioassay MUV-548.'
                }
            },
            'MUV-600': {
                'dataset_desc': 'Predict whether a molecule is active in bioassay MUV-600.',
                'label_dict': {0: 'inactive', 1: 'active'},
                'label_desc': {
                    0: 'inactive: the molecule is inactive in bioassay MUV-600.',
                    1: 'active: the molecule is active in bioassay MUV-600.'
                }
            },
            'MUV-644': {
                'dataset_desc': 'Predict whether a molecule is active in bioassay MUV-644.',
                'label_dict': {0: 'inactive', 1: 'active'},
                'label_desc': {
                    0: 'inactive: the molecule is inactive in bioassay MUV-644.',
                    1: 'active: the molecule is active in bioassay MUV-644.'
                }
            },
            'MUV-652': {
                'dataset_desc': 'Predict whether a molecule is active in bioassay MUV-652.',
                'label_dict': {0: 'inactive', 1: 'active'},
                'label_desc': {
                    0: 'inactive: the molecule is inactive in bioassay MUV-652.',
                    1: 'active: the molecule is active in bioassay MUV-652.'
                }
            },
            'MUV-689': {
                'dataset_desc': 'Predict whether a molecule is active in bioassay MUV-689.',
                'label_dict': {0: 'inactive', 1: 'active'},
                'label_desc': {
                    0: 'inactive: the molecule is inactive in bioassay MUV-689.',
                    1: 'active: the molecule is active in bioassay MUV-689.'
                }
            },
            'MUV-692': {
                'dataset_desc': 'Predict whether a molecule is active in bioassay MUV-692.',
                'label_dict': {0: 'inactive', 1: 'active'},
                'label_desc': {
                    0: 'inactive: the molecule is inactive in bioassay MUV-692.',
                    1: 'active: the molecule is active in bioassay MUV-692.'
                }
            },
            'MUV-712': {
                'dataset_desc': 'Predict whether a molecule is active in bioassay MUV-712.',
                'label_dict': {0: 'inactive', 1: 'active'},
                'label_desc': {
                    0: 'inactive: the molecule is inactive in bioassay MUV-712.',
                    1: 'active: the molecule is active in bioassay MUV-712.'
                }
            },
            'MUV-713': {
                'dataset_desc': 'Predict whether a molecule is active in bioassay MUV-713.',
                'label_dict': {0: 'inactive', 1: 'active'},
                'label_desc': {
                    0: 'inactive: the molecule is inactive in bioassay MUV-713.',
                    1: 'active: the molecule is active in bioassay MUV-713.'
                }
            },
            'MUV-733': {
                'dataset_desc': 'Predict whether a molecule is active in bioassay MUV-733.',
                'label_dict': {0: 'inactive', 1: 'active'},
                'label_desc': {
                    0: 'inactive: the molecule is inactive in bioassay MUV-733.',
                    1: 'active: the molecule is active in bioassay MUV-733.'
                }
            },
            'MUV-737': {
                'dataset_desc': 'Predict whether a molecule is active in bioassay MUV-737.',
                'label_dict': {0: 'inactive', 1: 'active'},
                'label_desc': {
                    0: 'inactive: the molecule is inactive in bioassay MUV-737.',
                    1: 'active: the molecule is active in bioassay MUV-737.'
                }
            },
            'MUV-810': {
                'dataset_desc': 'Predict whether a molecule is active in bioassay MUV-810.',
                'label_dict': {0: 'inactive', 1: 'active'},
                'label_desc': {
                    0: 'inactive: the molecule is inactive in bioassay MUV-810.',
                    1: 'active: the molecule is active in bioassay MUV-810.'
                }
            },
            'MUV-832': {
                'dataset_desc': 'Predict whether a molecule is active in bioassay MUV-832.',
                'label_dict': {0: 'inactive', 1: 'active'},
                'label_desc': {
                    0: 'inactive: the molecule is inactive in bioassay MUV-832.',
                    1: 'active: the molecule is active in bioassay MUV-832.'
                }
            },
            'MUV-846': {
                'dataset_desc': 'Predict whether a molecule is active in bioassay MUV-846.',
                'label_dict': {0: 'inactive', 1: 'active'},
                'label_desc': {
                    0: 'inactive: the molecule is inactive in bioassay MUV-846.',
                    1: 'active: the molecule is active in bioassay MUV-846.'
                }
            },
            'MUV-852': {
                'dataset_desc': 'Predict whether a molecule is active in bioassay MUV-852.',
                'label_dict': {0: 'inactive', 1: 'active'},
                'label_desc': {
                    0: 'inactive: the molecule is inactive in bioassay MUV-852.',
                    1: 'active: the molecule is active in bioassay MUV-852.'
                }
            },
            'MUV-858': {
                'dataset_desc': 'Predict whether a molecule is active in bioassay MUV-858.',
                'label_dict': {0: 'inactive', 1: 'active'},
                'label_desc': {
                    0: 'inactive: the molecule is inactive in bioassay MUV-858.',
                    1: 'active: the molecule is active in bioassay MUV-858.'
                }
            },
            'MUV-859': {
                'dataset_desc': 'Predict whether a molecule is active in bioassay MUV-859.',
                'label_dict': {0: 'inactive', 1: 'active'},
                'label_desc': {
                    0: 'inactive: the molecule is inactive in bioassay MUV-859.',
                    1: 'active: the molecule is active in bioassay MUV-859.'
                }
            }
        }
    },


    # ============================================================
    # SIDER: 27 side-effect groups
    # ============================================================
    # SIDER: Database of marketed drugs and adverse drug reactions (ADR), grouped into 27 system organ classes.
    # The Side Effect Resource (SIDER) is a database of marketed drugs and adverse drug reactions (ADR). The version of the SIDER dataset in DeepChem has grouped drug sideeffects into 27 system organ classes following MedDRA classifications measured for 1427 approved drugs (following previous usage).
    "sider": {
        "dataset_desc": (
            "Predict whether a marketed drug is associated with specific categories of adverse drug reactions (27 system organ classes)."
        ),
        "task_type": "multi_task_binary_classification",
        "tasks": {
            "Hepatobiliary disorders": {
                "dataset_desc": "Predict whether a drug causes side effects in the hepatobiliary system.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            },
            "Metabolism and nutrition disorders": {
                "dataset_desc": "Predict whether a drug causes metabolism and nutrition related side effects.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            },
            "Product issues": {
                "dataset_desc": "Predict whether a drug is associated with product-related issues as side effects.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            },
            "Eye disorders": {
                "dataset_desc": "Predict whether a drug causes eye-related side effects.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            },
            "Investigations": {
                "dataset_desc": "Predict whether a drug is associated with investigation-related adverse findings.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            },
            "Musculoskeletal and connective tissue disorders": {
                "dataset_desc": "Predict whether a drug causes musculoskeletal and connective tissue related side effects.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            },
            "Gastrointestinal disorders": {
                "dataset_desc": "Predict whether a drug causes gastrointestinal side effects.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            },
            "Social circumstances": {
                "dataset_desc": "Predict whether a drug is associated with side effects categorized under social circumstances.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            },
            "Immune system disorders": {
                "dataset_desc": "Predict whether a drug causes immune system related side effects.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            },
            "Reproductive system and breast disorders": {
                "dataset_desc": "Predict whether a drug causes reproductive system or breast related side effects.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            },
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)": {
                "dataset_desc": "Predict whether a drug is associated with neoplasm-related side effects, including cysts and polyps.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            },
            "General disorders and administration site conditions": {
                "dataset_desc": "Predict whether a drug causes general disorders or administration site related side effects.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            },
            "Endocrine disorders": {
                "dataset_desc": "Predict whether a drug causes endocrine system related side effects.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            },
            "Surgical and medical procedures": {
                "dataset_desc": "Predict whether a drug is associated with side effects categorized under surgical and medical procedures.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            },
            "Vascular disorders": {
                "dataset_desc": "Predict whether a drug causes vascular system related side effects.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            },
            "Blood and lymphatic system disorders": {
                "dataset_desc": "Predict whether a drug causes blood or lymphatic system related side effects.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            },
            "Skin and subcutaneous tissue disorders": {
                "dataset_desc": "Predict whether a drug causes skin or subcutaneous tissue related side effects.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            },
            "Congenital, familial and genetic disorders": {
                "dataset_desc": "Predict whether a drug is associated with congenital, familial or genetic disorder related side effects.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            },
            "Infections and infestations": {
                "dataset_desc": "Predict whether a drug is associated with infection or infestation related side effects.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            },
            "Respiratory, thoracic and mediastinal disorders": {
                "dataset_desc": "Predict whether a drug causes respiratory, thoracic or mediastinal side effects.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            },
            "Psychiatric disorders": {
                "dataset_desc": "Predict whether a drug causes psychiatric side effects.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            },
            "Renal and urinary disorders": {
                "dataset_desc": "Predict whether a drug causes renal or urinary system related side effects.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            },
            "Pregnancy, puerperium and perinatal conditions": {
                "dataset_desc": "Predict whether a drug is associated with side effects in pregnancy, puerperium or perinatal conditions.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            },
            "Ear and labyrinth disorders": {
                "dataset_desc": "Predict whether a drug causes ear or labyrinth related side effects.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            },
            "Cardiac disorders": {
                "dataset_desc": "Predict whether a drug causes cardiac side effects.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            },
            "Nervous system disorders": {
                "dataset_desc": "Predict whether a drug causes nervous system related side effects.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            },
            "Injury, poisoning and procedural complications": {
                "dataset_desc": "Predict whether a drug is associated with side effects categorized as injury, poisoning or procedural complications.",
                "label_dict": {
                    0: "no-side-effect",
                    1: "side-effect"
                },
                "label_desc": {
                    0: "no-side-effect: the drug is not reported to cause adverse reactions in this system organ class.",
                    1: "side-effect: the drug is reported to cause adverse reactions in this system organ class."
                }
            }
        }
    }
}
    
