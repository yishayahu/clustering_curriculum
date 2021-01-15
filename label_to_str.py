import os
label_to_str = {}
if os.environ["dataset_name"] == "imagenet":
    label_to_str = {0: 'kit_fox', 1: 'English_setter', 2: 'Siberian_husky', 3: 'Australian_terrier', 4: 'English_springer', 5: 'grey_whale', 6: 'lesser_panda', 7: 'Egyptian_cat', 8: 'ibex', 9: 'Persian_cat', 10: 'cougar', 11: 'gazelle', 12: 'porcupine', 13: 'sea_lion', 14: 'malamute', 15: 'badger', 16: 'Great_Dane', 17: 'Walker_hound', 18: 'Welsh_springer_spaniel', 19: 'whippet', 20: 'Scottish_deerhound', 21: 'killer_whale', 22: 'mink', 23: 'African_elephant', 24: 'Weimaraner', 25: 'soft-coated_wheaten_terrier', 26: 'Dandie_Dinmont', 27: 'red_wolf', 28: 'Old_English_sheepdog', 29: 'jaguar', 30: 'otterhound', 31: 'bloodhound', 32: 'Airedale', 33: 'hyena', 34: 'meerkat', 35: 'giant_schnauzer', 36: 'titi', 37: 'three-toed_sloth', 38: 'sorrel', 39: 'black-footed_ferret', 40: 'dalmatian', 41: 'black-and-tan_coonhound', 42: 'papillon', 43: 'skunk', 44: 'Staffordshire_bullterrier', 45: 'Mexican_hairless', 46: 'Bouvier_des_Flandres', 47: 'weasel', 48: 'miniature_poodle', 49: 'Cardigan', 50: 'malinois', 51: 'bighorn', 52: 'fox_squirrel', 53: 'colobus', 54: 'tiger_cat', 55: 'Lhasa', 56: 'impala', 57: 'coyote', 58: 'Yorkshire_terrier', 59: 'Newfoundland', 60: 'brown_bear', 61: 'red_fox', 62: 'Norwegian_elkhound', 63: 'Rottweiler', 64: 'hartebeest', 65: 'Saluki', 66: 'grey_fox', 67: 'schipperke', 68: 'Pekinese', 69: 'Brabancon_griffon', 70: 'West_Highland_white_terrier', 71: 'Sealyham_terrier', 72: 'guenon', 73: 'mongoose', 74: 'indri', 75: 'tiger', 76: 'Irish_wolfhound', 77: 'wild_boar', 78: 'EntleBucher', 79: 'zebra', 80: 'ram', 81: 'French_bulldog', 82: 'orangutan', 83: 'basenji', 84: 'leopard', 85: 'Bernese_mountain_dog', 86: 'Maltese_dog', 87: 'Norfolk_terrier', 88: 'toy_terrier', 89: 'vizsla', 90: 'cairn', 91: 'squirrel_monkey', 92: 'groenendael', 93: 'clumber', 94: 'Siamese_cat', 95: 'chimpanzee', 96: 'komondor', 97: 'Afghan_hound', 98: 'Japanese_spaniel', 99: 'proboscis_monkey', 100: 'guinea_pig', 101: 'white_wolf', 102: 'ice_bear', 103: 'gorilla', 104: 'borzoi', 105: 'toy_poodle', 106: 'Kerry_blue_terrier', 107: 'ox', 108: 'Scotch_terrier', 109: 'Tibetan_mastiff', 110: 'spider_monkey', 111: 'Doberman', 112: 'Boston_bull', 113: 'Greater_Swiss_Mountain_dog', 114: 'Appenzeller', 115: 'Shih-Tzu', 116: 'Irish_water_spaniel', 117: 'Pomeranian', 118: 'Bedlington_terrier', 119: 'warthog', 120: 'Arabian_camel', 121: 'siamang', 122: 'miniature_schnauzer', 123: 'collie', 124: 'golden_retriever', 125: 'Irish_terrier', 126: 'affenpinscher', 127: 'Border_collie', 128: 'hare', 129: 'boxer', 130: 'silky_terrier', 131: 'beagle', 132: 'Leonberg', 133: 'German_short-haired_pointer', 134: 'patas', 135: 'dhole', 136: 'baboon', 137: 'macaque', 138: 'Chesapeake_Bay_retriever', 139: 'bull_mastiff', 140: 'kuvasz', 141: 'capuchin', 142: 'pug', 143: 'curly-coated_retriever', 144: 'Norwich_terrier', 145: 'flat-coated_retriever', 146: 'hog', 147: 'keeshond', 148: 'Eskimo_dog', 149: 'Brittany_spaniel', 150: 'standard_poodle', 151: 'Lakeland_terrier', 152: 'snow_leopard', 153: 'Gordon_setter', 154: 'dingo', 155: 'standard_schnauzer', 156: 'hamster', 157: 'Tibetan_terrier', 158: 'Arctic_fox', 159: 'wire-haired_fox_terrier', 160: 'basset', 161: 'water_buffalo', 162: 'American_black_bear', 163: 'Angora', 164: 'bison', 165: 'howler_monkey', 166: 'hippopotamus', 167: 'chow', 168: 'giant_panda', 169: 'American_Staffordshire_terrier', 170: 'Shetland_sheepdog', 171: 'Great_Pyrenees', 172: 'Chihuahua', 173: 'tabby', 174: 'marmoset', 175: 'Labrador_retriever', 176: 'Saint_Bernard', 177: 'armadillo', 178: 'Samoyed', 179: 'bluetick', 180: 'redbone', 181: 'polecat', 182: 'marmot', 183: 'kelpie', 184: 'gibbon', 185: 'llama', 186: 'miniature_pinscher', 187: 'wood_rabbit', 188: 'Italian_greyhound', 189: 'lion', 190: 'cocker_spaniel', 191: 'Irish_setter', 192: 'dugong', 193: 'Indian_elephant', 194: 'beaver', 195: 'Sussex_spaniel', 196: 'Pembroke', 197: 'Blenheim_spaniel', 198: 'Madagascar_cat', 199: 'Rhodesian_ridgeback', 200: 'lynx', 201: 'African_hunting_dog', 202: 'langur', 203: 'Ibizan_hound', 204: 'timber_wolf', 205: 'cheetah', 206: 'English_foxhound', 207: 'briard', 208: 'sloth_bear', 209: 'Border_terrier', 210: 'German_shepherd', 211: 'otter', 212: 'koala', 213: 'tusker', 214: 'echidna', 215: 'wallaby', 216: 'platypus', 217: 'wombat', 218: 'revolver', 219: 'umbrella', 220: 'schooner', 221: 'soccer_ball', 222: 'accordion', 223: 'ant', 224: 'starfish', 225: 'chambered_nautilus', 226: 'grand_piano', 227: 'laptop', 228: 'strawberry', 229: 'airliner', 230: 'warplane', 231: 'airship', 232: 'balloon', 233: 'space_shuttle', 234: 'fireboat', 235: 'gondola', 236: 'speedboat', 237: 'lifeboat', 238: 'canoe', 239: 'yawl', 240: 'catamaran', 241: 'trimaran', 242: 'container_ship', 243: 'liner', 244: 'pirate', 245: 'aircraft_carrier', 246: 'submarine', 247: 'wreck', 248: 'half_track', 249: 'tank', 250: 'missile', 251: 'bobsled', 252: 'dogsled', 253: 'bicycle-built-for-two', 254: 'mountain_bike', 255: 'freight_car', 256: 'passenger_car', 257: 'barrow', 258: 'shopping_cart', 259: 'motor_scooter', 260: 'forklift', 261: 'electric_locomotive', 262: 'steam_locomotive', 263: 'amphibian', 264: 'ambulance', 265: 'beach_wagon', 266: 'cab', 267: 'convertible', 268: 'jeep', 269: 'limousine', 270: 'minivan', 271: 'Model_T', 272: 'racer', 273: 'sports_car', 274: 'go-kart', 275: 'golfcart', 276: 'moped', 277: 'snowplow', 278: 'fire_engine', 279: 'garbage_truck', 280: 'pickup', 281: 'tow_truck', 282: 'trailer_truck', 283: 'moving_van', 284: 'police_van', 285: 'recreational_vehicle', 286: 'streetcar', 287: 'snowmobile', 288: 'tractor', 289: 'mobile_home', 290: 'tricycle', 291: 'unicycle', 292: 'horse_cart', 293: 'jinrikisha', 294: 'oxcart', 295: 'bassinet', 296: 'cradle', 297: 'crib', 298: 'four-poster', 299: 'bookcase', 300: 'china_cabinet', 301: 'medicine_chest', 302: 'chiffonier', 303: 'table_lamp', 304: 'file', 305: 'park_bench', 306: 'barber_chair', 307: 'throne', 308: 'folding_chair', 309: 'rocking_chair', 310: 'studio_couch', 311: 'toilet_seat', 312: 'desk', 313: 'pool_table', 314: 'dining_table', 315: 'entertainment_center', 316: 'wardrobe', 317: 'Granny_Smith', 318: 'orange', 319: 'lemon', 320: 'fig', 321: 'pineapple', 322: 'banana', 323: 'jackfruit', 324: 'custard_apple', 325: 'pomegranate', 326: 'acorn', 327: 'hip', 328: 'ear', 329: 'rapeseed', 330: 'corn', 331: 'buckeye', 332: 'organ', 333: 'upright', 334: 'chime', 335: 'drum', 336: 'gong', 337: 'maraca', 338: 'marimba', 339: 'steel_drum', 340: 'banjo', 341: 'cello', 342: 'violin', 343: 'harp', 344: 'acoustic_guitar', 345: 'electric_guitar', 346: 'cornet', 347: 'French_horn', 348: 'trombone', 349: 'harmonica', 350: 'ocarina', 351: 'panpipe', 352: 'bassoon', 353: 'oboe', 354: 'sax', 355: 'flute', 356: 'daisy', 357: "yellow_lady's_slipper", 358: 'cliff', 359: 'valley', 360: 'alp', 361: 'volcano', 362: 'promontory', 363: 'sandbar', 364: 'coral_reef', 365: 'lakeside', 366: 'seashore', 367: 'geyser', 368: 'hatchet', 369: 'cleaver', 370: 'letter_opener', 371: 'plane', 372: 'power_drill', 373: 'lawn_mower', 374: 'hammer', 375: 'corkscrew', 376: 'can_opener', 377: 'plunger', 378: 'screwdriver', 379: 'shovel', 380: 'plow', 381: 'chain_saw', 382: 'cock', 383: 'hen', 384: 'ostrich', 385: 'brambling', 386: 'goldfinch', 387: 'house_finch', 388: 'junco', 389: 'indigo_bunting', 390: 'robin', 391: 'bulbul', 392: 'jay', 393: 'magpie', 394: 'chickadee', 395: 'water_ouzel', 396: 'kite', 397: 'bald_eagle', 398: 'vulture', 399: 'great_grey_owl', 400: 'black_grouse', 401: 'ptarmigan', 402: 'ruffed_grouse', 403: 'prairie_chicken', 404: 'peacock', 405: 'quail', 406: 'partridge', 407: 'African_grey', 408: 'macaw', 409: 'sulphur-crested_cockatoo', 410: 'lorikeet', 411: 'coucal', 412: 'bee_eater', 413: 'hornbill', 414: 'hummingbird', 415: 'jacamar', 416: 'toucan', 417: 'drake', 418: 'red-breasted_merganser', 419: 'goose', 420: 'black_swan', 421: 'white_stork', 422: 'black_stork', 423: 'spoonbill', 424: 'flamingo', 425: 'American_egret', 426: 'little_blue_heron', 427: 'bittern', 428: 'crane', 429: 'limpkin', 430: 'American_coot', 431: 'bustard', 432: 'ruddy_turnstone', 433: 'red-backed_sandpiper', 434: 'redshank', 435: 'dowitcher', 436: 'oystercatcher', 437: 'European_gallinule', 438: 'pelican', 439: 'king_penguin', 440: 'albatross', 441: 'great_white_shark', 442: 'tiger_shark', 443: 'hammerhead', 444: 'electric_ray', 445: 'stingray', 446: 'barracouta', 447: 'coho', 448: 'tench', 449: 'goldfish', 450: 'eel', 451: 'rock_beauty', 452: 'anemone_fish', 453: 'lionfish', 454: 'puffer', 455: 'sturgeon', 456: 'gar', 457: 'loggerhead', 458: 'leatherback_turtle', 459: 'mud_turtle', 460: 'terrapin', 461: 'box_turtle', 462: 'banded_gecko', 463: 'common_iguana', 464: 'American_chameleon', 465: 'whiptail', 466: 'agama', 467: 'frilled_lizard', 468: 'alligator_lizard', 469: 'Gila_monster', 470: 'green_lizard', 471: 'African_chameleon', 472: 'Komodo_dragon', 473: 'triceratops', 474: 'African_crocodile', 475: 'American_alligator', 476: 'thunder_snake', 477: 'ringneck_snake', 478: 'hognose_snake', 479: 'green_snake', 480: 'king_snake', 481: 'garter_snake', 482: 'water_snake', 483: 'vine_snake', 484: 'night_snake', 485: 'boa_constrictor', 486: 'rock_python', 487: 'Indian_cobra', 488: 'green_mamba', 489: 'sea_snake', 490: 'horned_viper', 491: 'diamondback', 492: 'sidewinder', 493: 'European_fire_salamander', 494: 'common_newt', 495: 'eft', 496: 'spotted_salamander', 497: 'axolotl', 498: 'bullfrog', 499: 'tree_frog', 500: 'tailed_frog', 501: 'whistle', 502: 'wing', 503: 'paintbrush', 504: 'hand_blower', 505: 'oxygen_mask', 506: 'snorkel', 507: 'loudspeaker', 508: 'microphone', 509: 'screen', 510: 'mouse', 511: 'electric_fan', 512: 'oil_filter', 513: 'strainer', 514: 'space_heater', 515: 'stove', 516: 'guillotine', 517: 'barometer', 518: 'rule', 519: 'odometer', 520: 'scale', 521: 'analog_clock', 522: 'digital_clock', 523: 'wall_clock', 524: 'hourglass', 525: 'sundial', 526: 'parking_meter', 527: 'stopwatch', 528: 'digital_watch', 529: 'stethoscope', 530: 'syringe', 531: 'magnetic_compass', 532: 'binoculars', 533: 'projector', 534: 'sunglasses', 535: 'loupe', 536: 'radio_telescope', 537: 'bow', 538: 'cannon', 539: 'assault_rifle', 540: 'rifle', 541: 'projectile', 542: 'computer_keyboard', 543: 'typewriter_keyboard', 544: 'crane', 545: 'lighter', 546: 'abacus', 547: 'cash_machine', 548: 'slide_rule', 549: 'desktop_computer', 550: 'hand-held_computer', 551: 'notebook', 552: 'web_site', 553: 'harvester', 554: 'thresher', 555: 'printer', 556: 'slot', 557: 'vending_machine', 558: 'sewing_machine', 559: 'joystick', 560: 'switch', 561: 'hook', 562: 'car_wheel', 563: 'paddlewheel', 564: 'pinwheel', 565: "potter's_wheel", 566: 'gas_pump', 567: 'carousel', 568: 'swing', 569: 'reel', 570: 'radiator', 571: 'puck', 572: 'hard_disc', 573: 'sunglass', 574: 'pick', 575: 'car_mirror', 576: 'solar_dish', 577: 'remote_control', 578: 'disk_brake', 579: 'buckle', 580: 'hair_slide', 581: 'knot', 582: 'combination_lock', 583: 'padlock', 584: 'nail', 585: 'safety_pin', 586: 'screw', 587: 'muzzle', 588: 'seat_belt', 589: 'ski', 590: 'candle', 591: "jack-o'-lantern", 592: 'spotlight', 593: 'torch', 594: 'neck_brace', 595: 'pier', 596: 'tripod', 597: 'maypole', 598: 'mousetrap', 599: 'spider_web', 600: 'trilobite', 601: 'harvestman', 602: 'scorpion', 603: 'black_and_gold_garden_spider', 604: 'barn_spider', 605: 'garden_spider', 606: 'black_widow', 607: 'tarantula', 608: 'wolf_spider', 609: 'tick', 610: 'centipede', 611: 'isopod', 612: 'Dungeness_crab', 613: 'rock_crab', 614: 'fiddler_crab', 615: 'king_crab', 616: 'American_lobster', 617: 'spiny_lobster', 618: 'crayfish', 619: 'hermit_crab', 620: 'tiger_beetle', 621: 'ladybug', 622: 'ground_beetle', 623: 'long-horned_beetle', 624: 'leaf_beetle', 625: 'dung_beetle', 626: 'rhinoceros_beetle', 627: 'weevil', 628: 'fly', 629: 'bee', 630: 'grasshopper', 631: 'cricket', 632: 'walking_stick', 633: 'cockroach', 634: 'mantis', 635: 'cicada', 636: 'leafhopper', 637: 'lacewing', 638: 'dragonfly', 639: 'damselfly', 640: 'admiral', 641: 'ringlet', 642: 'monarch', 643: 'cabbage_butterfly', 644: 'sulphur_butterfly', 645: 'lycaenid', 646: 'jellyfish', 647: 'sea_anemone', 648: 'brain_coral', 649: 'flatworm', 650: 'nematode', 651: 'conch', 652: 'snail', 653: 'slug', 654: 'sea_slug', 655: 'chiton', 656: 'sea_urchin', 657: 'sea_cucumber', 658: 'iron', 659: 'espresso_maker', 660: 'microwave', 661: 'Dutch_oven', 662: 'rotisserie', 663: 'toaster', 664: 'waffle_iron', 665: 'vacuum', 666: 'dishwasher', 667: 'refrigerator', 668: 'washer', 669: 'Crock_Pot', 670: 'frying_pan', 671: 'wok', 672: 'caldron', 673: 'coffeepot', 674: 'teapot', 675: 'spatula', 676: 'altar', 677: 'triumphal_arch', 678: 'patio', 679: 'steel_arch_bridge', 680: 'suspension_bridge', 681: 'viaduct', 682: 'barn', 683: 'greenhouse', 684: 'palace', 685: 'monastery', 686: 'library', 687: 'apiary', 688: 'boathouse', 689: 'church', 690: 'mosque', 691: 'stupa', 692: 'planetarium', 693: 'restaurant', 694: 'cinema', 695: 'home_theater', 696: 'lumbermill', 697: 'coil', 698: 'obelisk', 699: 'totem_pole', 700: 'castle', 701: 'prison', 702: 'grocery_store', 703: 'bakery', 704: 'barbershop', 705: 'bookshop', 706: 'butcher_shop', 707: 'confectionery', 708: 'shoe_shop', 709: 'tobacco_shop', 710: 'toyshop', 711: 'fountain', 712: 'cliff_dwelling', 713: 'yurt', 714: 'dock', 715: 'brass', 716: 'megalith', 717: 'bannister', 718: 'breakwater', 719: 'dam', 720: 'chainlink_fence', 721: 'picket_fence', 722: 'worm_fence', 723: 'stone_wall', 724: 'grille', 725: 'sliding_door', 726: 'turnstile', 727: 'mountain_tent', 728: 'scoreboard', 729: 'honeycomb', 730: 'plate_rack', 731: 'pedestal', 732: 'beacon', 733: 'mashed_potato', 734: 'bell_pepper', 735: 'head_cabbage', 736: 'broccoli', 737: 'cauliflower', 738: 'zucchini', 739: 'spaghetti_squash', 740: 'acorn_squash', 741: 'butternut_squash', 742: 'cucumber', 743: 'artichoke', 744: 'cardoon', 745: 'mushroom', 746: 'shower_curtain', 747: 'jean', 748: 'carton', 749: 'handkerchief', 750: 'sandal', 751: 'ashcan', 752: 'safe', 753: 'plate', 754: 'necklace', 755: 'croquet_ball', 756: 'fur_coat', 757: 'thimble', 758: 'pajama', 759: 'running_shoe', 760: 'cocktail_shaker', 761: 'chest', 762: 'manhole_cover', 763: 'modem', 764: 'tub', 765: 'tray', 766: 'balance_beam', 767: 'bagel', 768: 'prayer_rug', 769: 'kimono', 770: 'hot_pot', 771: 'whiskey_jug', 772: 'knee_pad', 773: 'book_jacket', 774: 'spindle', 775: 'ski_mask', 776: 'beer_bottle', 777: 'crash_helmet', 778: 'bottlecap', 779: 'tile_roof', 780: 'mask', 781: 'maillot', 782: 'Petri_dish', 783: 'football_helmet', 784: 'bathing_cap', 785: 'teddy', 786: 'holster', 787: 'pop_bottle', 788: 'photocopier', 789: 'vestment', 790: 'crossword_puzzle', 791: 'golf_ball', 792: 'trifle', 793: 'suit', 794: 'water_tower', 795: 'feather_boa', 796: 'cloak', 797: 'red_wine', 798: 'drumstick', 799: 'shield', 800: 'Christmas_stocking', 801: 'hoopskirt', 802: 'menu', 803: 'stage', 804: 'bonnet', 805: 'meat_loaf', 806: 'baseball', 807: 'face_powder', 808: 'scabbard', 809: 'sunscreen', 810: 'beer_glass', 811: 'hen-of-the-woods', 812: 'guacamole', 813: 'lampshade', 814: 'wool', 815: 'hay', 816: 'bow_tie', 817: 'mailbag', 818: 'water_jug', 819: 'bucket', 820: 'dishrag', 821: 'soup_bowl', 822: 'eggnog', 823: 'mortar', 824: 'trench_coat', 825: 'paddle', 826: 'chain', 827: 'swab', 828: 'mixing_bowl', 829: 'potpie', 830: 'wine_bottle', 831: 'shoji', 832: 'bulletproof_vest', 833: 'drilling_platform', 834: 'binder', 835: 'cardigan', 836: 'sweatshirt', 837: 'pot', 838: 'birdhouse', 839: 'hamper', 840: 'ping-pong_ball', 841: 'pencil_box', 842: 'pay-phone', 843: 'consomme', 844: 'apron', 845: 'punching_bag', 846: 'backpack', 847: 'groom', 848: 'bearskin', 849: 'pencil_sharpener', 850: 'broom', 851: 'mosquito_net', 852: 'abaya', 853: 'mortarboard', 854: 'poncho', 855: 'crutch', 856: 'Polaroid_camera', 857: 'space_bar', 858: 'cup', 859: 'racket', 860: 'traffic_light', 861: 'quill', 862: 'radio', 863: 'dough', 864: 'cuirass', 865: 'military_uniform', 866: 'lipstick', 867: 'shower_cap', 868: 'monitor', 869: 'oscilloscope', 870: 'mitten', 871: 'brassiere', 872: 'French_loaf', 873: 'vase', 874: 'milk_can', 875: 'rugby_ball', 876: 'paper_towel', 877: 'earthstar', 878: 'envelope', 879: 'miniskirt', 880: 'cowboy_hat', 881: 'trolleybus', 882: 'perfume', 883: 'bathtub', 884: 'hotdog', 885: 'coral_fungus', 886: 'bullet_train', 887: 'pillow', 888: 'toilet_tissue', 889: 'cassette', 890: "carpenter's_kit", 891: 'ladle', 892: 'stinkhorn', 893: 'lotion', 894: 'hair_spray', 895: 'academic_gown', 896: 'dome', 897: 'crate', 898: 'wig', 899: 'burrito', 900: 'pill_bottle', 901: 'chain_mail', 902: 'theater_curtain', 903: 'window_shade', 904: 'barrel', 905: 'washbasin', 906: 'ballpoint', 907: 'basketball', 908: 'bath_towel', 909: 'cowboy_boot', 910: 'gown', 911: 'window_screen', 912: 'agaric', 913: 'cellular_telephone', 914: 'nipple', 915: 'barbell', 916: 'mailbox', 917: 'lab_coat', 918: 'fire_screen', 919: 'minibus', 920: 'packet', 921: 'maze', 922: 'pole', 923: 'horizontal_bar', 924: 'sombrero', 925: 'pickelhaube', 926: 'rain_barrel', 927: 'wallet', 928: 'cassette_player', 929: 'comic_book', 930: 'piggy_bank', 931: 'street_sign', 932: 'bell_cote', 933: 'fountain_pen', 934: 'Windsor_tie', 935: 'volleyball', 936: 'overskirt', 937: 'sarong', 938: 'purse', 939: 'bolo_tie', 940: 'bib', 941: 'parachute', 942: 'sleeping_bag', 943: 'television', 944: 'swimming_trunks', 945: 'measuring_cup', 946: 'espresso', 947: 'pizza', 948: 'breastplate', 949: 'shopping_basket', 950: 'wooden_spoon', 951: 'saltshaker', 952: 'chocolate_sauce', 953: 'ballplayer', 954: 'goblet', 955: 'gyromitra', 956: 'stretcher', 957: 'water_bottle', 958: 'dial_telephone', 959: 'soap_dispenser', 960: 'jersey', 961: 'school_bus', 962: 'jigsaw_puzzle', 963: 'plastic_bag', 964: 'reflex_camera', 965: 'diaper', 966: 'Band_Aid', 967: 'ice_lolly', 968: 'velvet', 969: 'tennis_ball', 970: 'gasmask', 971: 'doormat', 972: 'Loafer', 973: 'ice_cream', 974: 'pretzel', 975: 'quilt', 976: 'maillot', 977: 'tape_player', 978: 'clog', 979: 'iPod', 980: 'bolete', 981: 'scuba_diver', 982: 'pitcher', 983: 'matchstick', 984: 'bikini', 985: 'sock', 986: 'CD_player', 987: 'lens_cap', 988: 'thatch', 989: 'vault', 990: 'beaker', 991: 'bubble', 992: 'cheeseburger', 993: 'parallel_bars', 994: 'flagpole', 995: 'coffee_mug', 996: 'rubber_eraser', 997: 'stole', 998: 'carbonara', 999: 'dumbbell'}
elif os.environ["dataset_name"] == "cifar10":
    label_to_str = {0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}
elif os.environ["dataset_name"] == "tiny_imagenet":
    label_to_str = {}
else:
    raise Exception("dataset does not exists")