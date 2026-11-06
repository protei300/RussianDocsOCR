"""Registration page reader for the service: page -> stamps -> fields of each stamp.

``page.RegistrationReader`` finds and cuts the stamps with the library (it is not
modified); ``stamp.StampReader`` reads one cut stamp with the field network, the
stamp OCR v3. The result preserves the five detector classes and raw OCR text.
"""
