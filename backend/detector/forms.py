from django import forms

class UploadImageForm(forms.Form):
    image = forms.ImageField(label='Upload Skin Image')