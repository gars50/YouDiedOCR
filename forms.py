from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, BooleanField, SubmitField, validators, SelectField


class InputVideoForm(FlaskForm):
    youtube_url = StringField('URL of the Youtube video to process', validators=[validators.DataRequired()], default='https://www.youtube.com/watch?v=aUANPtlh7pw')
    game = SelectField('Game', choices=[('ring','Elden Ring'), ('sekiro','Sekiro'), ('souls',"Dark Souls 1/2/3 or Demon's Souls Remake")], default='ring')
    debug = BooleanField('Run this in debug.', default=False)
    milliseconds_skipped = IntegerField('How many milliseconds between each frame to check (0.2 seconds works well, 200ms)', validators=[validators.NumberRange(min=0)], default=200)
    seconds_skipped_after_death = IntegerField('Skip X seconds after finding a death', validators=[validators.NumberRange(min=0)], default=15)

    submit = SubmitField('Process')