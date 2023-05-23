from waitress import serve
from app import app

serve(app, host='192.168.21.77', port=5000)
