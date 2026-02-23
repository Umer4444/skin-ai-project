from django.shortcuts import render
from .forms import UploadImageForm
from django.conf import settings
import os

def home(request):
    uploaded_file_url = None
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            save_path = os.path.join(settings.MEDIA_ROOT, 'uploads', image.name)
            
            with open(save_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)
            
            uploaded_file_url = f"/media/uploads/{image.name}"
    else:
        form = UploadImageForm()
    
    return render(request, 'home.html', {
        'form': form,
        'uploaded_file_url': uploaded_file_url
    })