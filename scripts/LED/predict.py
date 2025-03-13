from transformers import LEDTokenizer, LEDForConditionalGeneration
import re


# Load the fine-tuned model and tokenizer
model_path = "../results/LED/final_model"
model = LEDForConditionalGeneration.from_pretrained(model_path)
tokenizer = LEDTokenizer.from_pretrained(model_path)


# Example input
input_text = """
Extract the following fields from the invoice:
- Invoice Number
- Invoice Date
- Supplier Name
- Total Amount
- VAT Amount
- List of Products (with fields: Product ID, Name, Quantity, Cost, VAT Rate)

Invoice Text:

GI.GROUP SRL

VIALE PARIOLI,44 - 00197 Roma (RM) - Italy

C.F./P.lva 15474631007 Reg. imprese 1592948

IN Tel. 392/4563011 / Sede Operativa - Via di San Romano,56 A/B
= e-mail: infogigroupsri@gmail.com Pec: gigroupsrl@legalmail.it_ Internet: https://www.gigroupstore.com

Ingrosso di carta e distribuzione di materiale monouso e detergenza professionale HACCP.

Fattura nr. 1371 del | 02/12/2024
Destinatario Destinazione
FARO S.R.L.S. FARO S.R.L.S.
VIA PIAVE, 55 VIA PIAVE, 55
00187 ROMA (RM) 00187 ROMA (RM)
ITALY ITALY
C.F./P.lva 13796331000
Codice Descrizione Quantita Prezzo Sconto Importo Iva
Rif. Doc. di trasporto 882 del 06/11/2024, 911 del
13/11/2024, 910 del 14/11/2024, 930 del 21/11/2024, 952
del 27/11/2024:
91112 BICCHIERE CARTA L/CALDO 120Z- 360ML -CONF 50PZ 8confezione € 4,500 €36,00 22
82994000 TOVAGLIOLO 38X38 ECONATURAL - MICROCOLLATO - 90confezione € 1,000 €90,00 22
CONFEZIONE 40PZ - CARTONE 30 CONF
91110AL BICCHIERE CART.BIO 80Z-240 ML- CONF 50 PZ- 14confezione € 2,550 €35,70 22
CARTONE 16 CONF
82228600 TOVAGLIOLO 24X24 DRINK ECONATURAL - 72confezione € 2,500 €180,00 22
FIBERPACK - CONFEZIONE 100PZ - CARTONE 12 CONF
99154 COPERCHIO CARTA BICCHIERE 802-240 ML - CONF Qconfezione € 2,550 €22,95 22
50PZ S/BECCUCCIO
32620 TOVAGLIOLO NERO 25X25-2 VELI-CARTONE 20 CONF 40confezione € 2,100 €84,00 22
91110AL BICCHIERE CART.BIO 80Z-240 ML- CONF 50 PZ- 20confezione € 2,500 €50,00 22
CARTONE 16 CONF
99112 COPERCHIO BICCHIERE 120Z-360 ML-CONF 50PZ 3confezione € 5,000 €15,00 22
91115 BICCHIERE CART.BIO 70Z-200 ML- CONF 50 PZ- 20confezione € 2,000 €40,00 22
CARTONE 20 CONF
90223 BIS POSATE IN LEGNO - WE BIO - TOV.33X33 - 1cartone €55,000 €55,00 22
CARTONE 500PZ
91112 BICCHIERE CARTA L/CALDO 120Z- 360ML -CONF 50PZ 3confezione € 4,000 €12,00 22
99154 COPERCHIO CARTA BICCHIERE 802-240 ML - CONF Aconfezione € 3,000 €12,00 22
50PZ S/BECCUCCIO
0097 FOODBOX SMART FOLDABLE - PAPER KRAFT 1cartone €65,000 €65,00 22
152X120X65 MM - CARTONE 300 PZ
Copia della fattura elettronica disponibile nella Sua area
riservata dell’Agenzia delle Entrate
Iva Imponibile Imposta
22: Imponibile 22% € 697,65 € 153,48
Pagamento: Bonifico vista fattura Tot. imponibile € 697,65
INTESA SAN PAOLO S.P.A Tot. Iva € 153,48
IBAN IT 67 R 03069 03202 100000067680
Scadenze: 02/12/2024 € 851,13
Tot. documento € 851,13

Nel rispetto dalla normativa vigente, ivi incluso DL 196/03 e reg. UE 2016/679, informiamo che i Vs. dati saranno utilizzati ai soli fini connessi ai rapporti commerciali tra di noi in essere.
Contributo CONAI assolto ove dovuto - Vi preghiamo di controllare i Vs. dati anagrafici, la P. |VA e il Cod. Fiscale. Non ci riteniamo responsabili di eventuali errori.

Pag. 1"""

# Tokenize input
input_ids = tokenizer(
    f"{input_text}",
    return_tensors="pt",
    max_length=16384,  # LED supports up to 16,384 tokens
    truncation=True,  # Truncate if necessary
).input_ids

# Generate output
outputs = model.generate(input_ids)
predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(predicted_text)

# Parse the predicted output
def parse_predicted_output(predicted_text):
    result = {}
    # Extract fields using regex
    result["id"] = re.search(r"id: (.+?),", predicted_text).group(1)
    result["date"] = re.search(r"date: (.+?),", predicted_text).group(1)
    result["supplier"] = re.search(r"supplier: (.+?),", predicted_text).group(1)
    result["amount"] = float(re.search(r"amount: (.+?),", predicted_text).group(1))
    result["vat_amount"] = float(re.search(r"vat_amount: (.+?),", predicted_text).group(1))

    # Extract products
    products_text = re.search(r"products: \[(.+)\]", predicted_text).group(1)
    products = []
    for product_text in products_text.split("}, "):
        product = {}
        product["id"] = re.search(r"id: (.+?),", product_text).group(1)
        product["name"] = re.search(r"name: (.+?),", product_text).group(1)
        product["quantity"] = int(re.search(r"quantity: (.+?),", product_text).group(1))
        product["cost"] = float(re.search(r"cost: (.+?),", product_text).group(1))
        product["vat_rate"] = int(re.search(r"vat_rate: (.+?)[,}]", product_text).group(1))
        products.append(product)
    result["products"] = products

    return result


# Example usage
#parsed_output = parse_predicted_output(predicted_text)
#print(parsed_output)