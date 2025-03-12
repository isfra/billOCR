from transformers import pipeline

# Load the fine-tuned model
invoice_parser = pipeline("token-classification", model="./results")

# Example invoice text
invoice_text = """
Fattura nr. 1341 del 02/11/2024
Destinatario: FARO S.R.L.S. VIA PIAVE, 55 00187 ROMA (RM) ITALY
Codice: 90223 | Descrizione: BIS POSATE IN LEGNO - WE BIO - TOV.33X33 CARTONE 50DPZ | Quantità: 1 | Prezzo: €10.00 | Importo: €10.00
Totale Imponibile: €1,061.40 | IVA: €233.51 | Totale: €1,294.91
"""

# Parse the invoice
results = invoice_parser(invoice_text)
print(results)