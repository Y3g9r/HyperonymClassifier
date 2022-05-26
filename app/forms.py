from flask_wtf import FlaskForm
from wtforms import StringField, BooleanField
from wtforms.validators import InputRequired

class LoginForm(FlaskForm):
    openid = StringField('openid', validators = [InputRequired()])
    remember_me = BooleanField('remember_me', default = False)