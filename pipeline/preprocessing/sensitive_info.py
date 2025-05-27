'''
Basic redacting info using regular expressions
'''

import re

class Redact:
    def __init__(self):
        self = self
        
    def redact_sensitive(text: str) -> str:
        #text = re.sub(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', '[REDACTED_EMAIL]', text, flags = re.I)
        #text = re.sub(r'\b\d{3}[-.\s]??\d{2,3}[-.\s]??\d{4}\b', '[REDACTED_PHONE]', text)
        text = re.sub(r'\b(SSN|DOB|Salary|Confidential|Private)\b.*', '[REDACTED]', text, flags = re.I)
    
        return text