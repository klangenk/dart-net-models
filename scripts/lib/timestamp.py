from datetime import datetime

def timestamp():
  return datetime.now().strftime('%y-%m-%d--%H-%M-%S-%f')