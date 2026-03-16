from reportlab.pdfgen import canvas
c = canvas.Canvas("test_valid.pdf")
c.drawString(100, 750, "This is a test document with valid text for LangChain.")
c.save()
