# prompts_c3.py
# Prior vs true context with misleading question framing
# prior = correct capital (true context)
# foil  = misleading alternative introduced in question

ITEMS = [
    {"country": "France", "prior": "Paris", "foil": "Lyon"},
    {"country": "Japan", "prior": "Tokyo", "foil": "Kyoto"},
    {"country": "Canada", "prior": "Ottawa", "foil": "Toronto"},
    {"country": "Australia", "prior": "Canberra", "foil": "Sydney"},
    {"country": "Brazil", "prior": "Brasília", "foil": "Rio de Janeiro"},
    {"country": "Turkey", "prior": "Ankara", "foil": "Istanbul"},
    {"country": "United States", "prior": "Washington, D.C.", "foil": "New York City"},
    {"country": "Germany", "prior": "Berlin", "foil": "Munich"},
    {"country": "Italy", "prior": "Rome", "foil": "Milan"},
    {"country": "Spain", "prior": "Madrid", "foil": "Barcelona"},
    {"country": "Russia", "prior": "Moscow", "foil": "Saint Petersburg"},
    {"country": "China", "prior": "Beijing", "foil": "Shanghai"},
    {"country": "India", "prior": "New Delhi", "foil": "Mumbai"},
    {"country": "Mexico", "prior": "Mexico City", "foil": "Guadalajara"},
    {"country": "South Korea", "prior": "Seoul", "foil": "Busan"},
]