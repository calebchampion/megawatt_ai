import re
from typing import List, Dict

class Names:
  def __init__(self, docs: List[Dict]):
    self.docs = docs

  def replace_IDs_w_names(self):
    #hardcoded for privacy
    user_map = {
      'U018LP1LWQM': 'Ilya',
      'U030YUU2VED': 'Dean',
      'U03N6ENPG8Y': 'Wayne',
      'U04753AP11V': 'Patrick',
      'U072K6L9GD8': 'Caleb',
      'U06AN42RUF9': 'Mason',
      'U05QF720YF4': 'Tristan',
      'U08QC4568Q5': 'Steven',
      'U08T7KADB7B': 'Rudy', 
      'U0898JBANP6': 'Jason',
      'U030P58UAGK': 'Arniel', 
    }

    def replace_mentions(text):
      return re.sub(r"<@([A-Z0-9]+)>", lambda m: f"@{user_map.get(m.group(1), m.group(1))}", text)

    for doc in self.docs:
      doc['text'] = replace_mentions(doc['text'])  #replace mentions in text
      doc['user'] = user_map.get(doc['user'], doc['user'])  #replace user field if unkown
      