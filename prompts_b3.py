# prompts_b3.py
# B3: Question-before-document ordering baseline
# prior = true capital
# ctx   = false capital asserted by the document (same as C1 foils)

ITEMS = [
    {"country": "France", "prior": "Paris", "ctx": "Lyon"},
    {"country": "Japan", "prior": "Tokyo", "ctx": "Kyoto"},
    {"country": "Canada", "prior": "Ottawa", "ctx": "Toronto"},
    {"country": "Australia", "prior": "Canberra", "ctx": "Sydney"},
    {"country": "Brazil", "prior": "Brasília", "ctx": "Rio de Janeiro"},
    {"country": "Turkey", "prior": "Ankara", "ctx": "Istanbul"},
    {"country": "United States", "prior": "Washington, D.C.", "ctx": "New York City"},
    {"country": "Germany", "prior": "Berlin", "ctx": "Munich"},
    {"country": "Italy", "prior": "Rome", "ctx": "Milan"},
    {"country": "Spain", "prior": "Madrid", "ctx": "Barcelona"},
    {"country": "Russia", "prior": "Moscow", "ctx": "Saint Petersburg"},
    {"country": "China", "prior": "Beijing", "ctx": "Shanghai"},
    {"country": "India", "prior": "New Delhi", "ctx": "Mumbai"},
    {"country": "Mexico", "prior": "Mexico City", "ctx": "Guadalajara"},
    {"country": "South Korea", "prior": "Seoul", "ctx": "Busan"},
]