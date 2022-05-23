from app import app

@app.route('/')
@app.route('/index')
def index():
    return "My first flask web"