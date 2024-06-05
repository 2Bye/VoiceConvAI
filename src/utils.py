import re


def extract_soap_sections(text):
    pattern = re.compile(
        r"(S:.*?)(?=O:|A:|P:|$)|(O:.*?)(?=S:|A:|P:|$)|(A:.*?)(?=S:|O:|P:|$)|(P:.*?)(?=S:|O:|A:|$)", re.DOTALL
    )

    matches = pattern.findall(text)

    filtered_note = " ".join(filter(None, [item for sublist in matches for item in sublist]))

    return filtered_note
