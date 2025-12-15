"""TXT parser placeholder."""
def parse(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [{"text": f.read()}]
