from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class LHCDataset(XMLDataset):
    # CLASSES = ("Baby diapers",
    #            "Baby Furniture",
    #            "Baby washing and nursing supplie",
    #            "Baby slippers",
    #            "Baby handkerchiefs ",
    #            "Baby crib",
    #            "Baby carriage",
    #            "Baby tableware",
    #            "Dairy",
    #            "Cocktail",
    #            "Red wine",
    #            "Liquor and Spirits",
    #            "Carbonated drinks",
    #            "Herbal tea",
    #            "Coffee",
    #            "Tea beverage",
    #            "Baby milk powder",
    #            "Guozhen",
    #            # "Ginger Tea",             # DeleteClass 20191102
    #            # "Sour Plum Soup",            # DeleteClass 20191102  20
    #            "Adult milk powder",
    #            "Tea",
    #            "Notebook",
    #            "Pencil case",
    #            "Pen",
    #            "Baby Toys",
    #            "Children Toys",
    #            "Football",
    #            #"Rubber ball",           # DeleteClass 20191103
    #            "Badminton",
    #            "Basketball",
    #            "Skate",
    #            "Pasta",
    #            "Noodle",
    #            "Flour",
    #            "Rise",
    #            "Oats",
    #            "Sesame paste",
    #            "Soymilk",
    #            #"Lotus root flour",          # DeleteClass 20191103
    #            "Walnut powder",
    #            "Quick-frozen Tangyuan",
    #            "Quick-frozen Wonton",
    #            "Quick-frozen dumplings",
    #            "Can",
    #            "Instant noodles",
    #            "Mixed congee",
    #            "Potato chips",
    #            "Dried meat",
    #            #"Chicken claws",             # DeleteClass 20191103
    #            #"Hot strips",             # DeleteClass 20191103
    #            #"Dried fish",             # DeleteClass 20191103
    #            #"Dried beans",             # DeleteClass 20191103
    #            #"Fish tofu",             # DeleteClass 20191103
    #            "Chocolates",
    #            "Chewing gum",
    #            "Cake",
    #            "Pie",
    #            "Biscuits",
    #            # "Potatoes",    # DeleteClass 20191102
    #            "Ice cream",
    #            "Cooking wine",
    #            "Soy sauce",
    #            "Sauce",
    #            "Vinegar",
    #            "Care Kit",
    #            "Shampoo",
    #            "Hair conditioner",
    #            "Hair gel",
    #            "Hair dye",
    #            "Comb",
    #            "Tampon",
    #            "Cotton swab",
    #            "Band aid",
    #            "Adult Diapers",
    #            "Bath lotion",
    #            # "Soap",                 # DeleteClass 20191102
    #            # "Flower dew",           # DeleteClass 20191102  10
    #            "Emulsion",
    #            "Facial Cleanser",
    #            "Razor",
    #            "Facial mask",
    #            "Skin care set",
    #            "Toothbrush",
    #            # "Dental floss bar",        # DeleteClass 20191102
    #            "Toothpaste",
    #            "Mouth wash",
    #            "Makeup tools",
    #            "Jacket",
    #            "Trousers",
    #            "Adult shoes",
    #            "Adult socks",
    #            "Children shoes",
    #            "Children Socks",
    #            "Children hats",
    #            #"Children underwear",             # DeleteClass 20191103
    #            "Lingerie",
    #            "Men underwear",
    #            "Adult hat",
    #            "Bedding set",
    #            "Juicer",
    #            "Washing machine",
    #            "Microwave Oven",
    #            "Desk lamp",
    #            "Air conditioning fan",
    #            "Air conditioner",
    #            "Soybean Milk machine",
    #            "Electric iron",
    #            "Electric kettle",
    #            # "Pressure cooker",            # DeleteClass 20191102
    #            "Television",
    #            "Electric Hot pot",
    #            "Electric fan",
    #            "Rice cooker",
    #            "Electromagnetic furnace",
    #            "Electric frying pan",
    #            #"Electric steaming pan",             # DeleteClass 20191103
    #            "Hair drier",
    #            "Socket",
    #            "Refrigerator",
    #            "Coat hanger",
    #            "Sports cup",
    #            "Disposable cups",
    #            "Thermos bottle",
    #            "Basin",
    #            "Mug",
    #            "Draw bar box",
    #            "Trash",
    #            "Disposable bag",
    #            "Storage box",
    #            "Storage bottle",
    #            "Stool",     # NewClass20191010
    #            # "Package",     # DeleteClass 20191102
    #            "Knapsack",
    #            "Chopping block",
    #            "Food box",
    #            #"Spoon",             # DeleteClass 20191103
    #            "Chopsticks",
    #            "Bowl",     # NewClass20191010
    #            "Pot shovel",
    #            "Soup ladle",
    #            #"Cutter",             # DeleteClass 20191103
    #            # "Oil brush", #DeleteClass20191027
    #            # "Knives",       # DeleteClass 20191102
    #            "Forks",
    #            "Dinner plate",
    #            "Fresh-keeping film"
    #            # "Fresh-keeping bag"   # DeleteClass 20191102
    #            )
    CLASSES = ("Baby diapers",
               "Baby Furniture",
               "Baby washing and nursing supplie",
               "Baby slippers",
               "Baby handkerchiefs ",
               "Baby crib",
               "Baby carriage",
               "Baby tableware",
               "Dairy",
               "Cocktail",
               "Red wine",
               "Liquor and Spirits",
               "Carbonated drinks",
               "Herbal tea",
               "Coffee",
               "Tea beverage",
               "Baby milk powder",
               "Guozhen",
               "Ginger Tea",             # DeleteClass 20191102
               "Sour Plum Soup",            # DeleteClass 20191102  20
               "Adult milk powder",
               "Tea",
               "Notebook",
               "Pencil case",
               "Pen",
               "Baby Toys",
               "Children Toys",
               "Football",
               "Rubber ball",           # DeleteClass 20191103
               "Badminton",
               "Basketball",
               "Skate",
               "Pasta",
               "Noodle",
               "Flour",
               "Rise",
               "Oats",
               "Sesame paste",
               "Soymilk",
               "Lotus root flour",          # DeleteClass 20191103
               "Walnut powder",
               "Quick-frozen Tangyuan",
               "Quick-frozen Wonton",
               "Quick-frozen dumplings",
               "Can",
               "Instant noodles",
               "Mixed congee",
               "Potato chips",
               "Dried meat",
               #"Chicken claws",             # DeleteClass 20191103           20200114Del
               "Hot strips",             # DeleteClass 20191103
               "Dried fish",             # DeleteClass 20191103
               "Dried beans",             # DeleteClass 20191103
               "Fish tofu",             # DeleteClass 20191103
               "Chocolates",
               "Chewing gum",
               "Cake",
               "Pie",
               "Biscuits",
               # "Potatoes",    # DeleteClass 20191102           20200114Del
               "Ice cream",
               "Cooking wine",
               "Soy sauce",
               "Sauce",
               "Vinegar",
               "Care Kit",
               "Shampoo",
               "Hair conditioner",
               "Hair gel",
               "Hair dye",
               "Comb",
               "Tampon",
               "Cotton swab",
               "Band aid",
               "Adult Diapers",
               "Bath lotion",
               "Soap",                 # DeleteClass 20191102
               # "Flower dew",           # DeleteClass 20191102  10       20200114Del
               "Emulsion",
               "Facial Cleanser",
               "Razor",
               "Facial mask",
               "Skin care set",
               "Toothbrush",
               # "Dental floss bar",        # DeleteClass 20191102       20200114Del
               "Toothpaste",
               "Mouth wash",
               "Makeup tools",
               "Jacket",
               "Trousers",
               "Adult shoes",
               "Adult socks",
               "Children shoes",
               "Children Socks",
               "Children hats",
               "Children underwear",             # DeleteClass 20191103
               "Lingerie",
               "Men underwear",
               "Adult hat",
               "Bedding set",
               "Juicer",
               "Washing machine",
               "Microwave Oven",
               "Desk lamp",
               "Air conditioning fan",
               "Air conditioner",
               "Soybean Milk machine",
               "Electric iron",
               "Electric kettle",
               # "Pressure cooker",            # DeleteClass 20191102       20200114Del
               "Television",
               "Electric Hot pot",
               "Electric fan",
               "Rice cooker",
               "Electromagnetic furnace",
               "Electric frying pan",
               "Electric steaming pan",             # DeleteClass 20191103
               "Hair drier",
               "Socket",
               "Refrigerator",
               "Coat hanger",
               "Sports cup",
               "Disposable cups",
               "Thermos bottle",
               "Basin",
               "Mug",
               "Draw bar box",
               "Trash",
               "Disposable bag",
               "Storage box",
               "Storage bottle",
               "Stool",     # NewClass20191010
               # "Package",     # DeleteClass 20191102       20200114Del
               "Knapsack",
               "Chopping block",
               "Food box",
               "Spoon",             # DeleteClass 20191103
               "Chopsticks",
               "Bowl",     # NewClass20191010
               "Pot shovel",
               "Soup ladle",
               "Cutter",             # DeleteClass 20191103
               # "Oil brush", #DeleteClass20191027
               "Knives",       # DeleteClass 20191102
               "Forks",
               "Dinner plate",
               "Fresh-keeping film"
               # "Fresh-keeping bag"   # DeleteClass 20191102               20200114Del
               )