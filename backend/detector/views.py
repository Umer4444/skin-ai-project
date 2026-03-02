from django.shortcuts import render
from .forms import UploadImageForm
from django.conf import settings
import os
from .model_utils import predict_skin_cancer

def home(request):
    uploaded_file_url = None
    label = None
    confidence = None
    heatmap_url = None
    overlay_url = None

    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)

        if form.is_valid():
            image = form.cleaned_data['image']

            # -------- SAVE IMAGE --------
            upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
            os.makedirs(upload_dir, exist_ok=True)

            save_path = os.path.join(upload_dir, image.name)

            with open(save_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            uploaded_file_url = f"/media/uploads/{image.name}"

            # -------- AI + GRADCAM --------
            label, confidence_score, heatmap_path, overlay_path = predict_skin_cancer(save_path)

            confidence = round(confidence_score * 100, 2)

            heatmap_url = heatmap_path.replace(settings.MEDIA_ROOT, "/media")
            overlay_url = overlay_path.replace(settings.MEDIA_ROOT, "/media")

    else:
        form = UploadImageForm()

    return render(request, 'home.html', {
        'form': form,
        'uploaded_file_url': uploaded_file_url,
        'label': label,
        'confidence': confidence,
        'heatmap_url': heatmap_url,
        'overlay_url': overlay_url
    })