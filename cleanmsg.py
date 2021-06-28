#!/usr/bin/env python
# coding: utf-8

# In[ ]:
def clean_message(text):
    import re
    result = re.sub(r"\S*.com\b","", text)
    result = re.findall(r'\b[^\d\W]+\b', result)
    result= " ".join(result)
    result = re.sub(r"\S*_","", result)
    return result

