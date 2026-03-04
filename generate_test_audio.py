from gtts import gTTS

texto = """
Boa tarde doutor. Estou sentindo dores pélvicas intensas há cerca de duas semanas.
A dor piora durante o ciclo menstrual e tenho tido sangramentos irregulares.
Também sinto muita fadiga e enjoo frequente.
Tenho histórico familiar de endometriose e estou preocupada com esses sintomas.
"""

tts = gTTS(text=texto, lang='pt', slow=False)
tts.save("data/samples/consulta_teste.mp3")
print("Audio gerado!")
