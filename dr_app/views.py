from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import os
import cv2
from .forms import ImageUploadForm
from .utils import apply_kirsch_filter, predict_image

def upload_image(request):
    prediction = None
    original_image_url = None
    kirsch_image_url = None

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)

        if form.is_valid():
            uploaded_image = form.cleaned_data['image']

            # Create upload directories if they don't exist
            upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploaded_images')
            processed_dir = os.path.join(settings.MEDIA_ROOT, 'processed_images')
            os.makedirs(upload_dir, exist_ok=True)
            os.makedirs(processed_dir, exist_ok=True)

            # Save original image
            filename = uploaded_image.name
            uploaded_image_path = os.path.join(upload_dir, filename)

            with open(uploaded_image_path, 'wb+') as destination:
                for chunk in uploaded_image.chunks():
                    destination.write(chunk)

            # Set original image URL for template
            original_image_url = settings.MEDIA_URL + 'uploaded_images/' + filename

            # Apply Kirsch filter and save processed image
            kirsch_output = apply_kirsch_filter(uploaded_image_path)
            kirsch_filename = f"processed_{filename}"
            kirsch_path = os.path.join(processed_dir, kirsch_filename)
            cv2.imwrite(kirsch_path, kirsch_output)

            # Set Kirsch image URL for template
            kirsch_image_url = settings.MEDIA_URL + 'processed_images/' + kirsch_filename

            # Predict using model
            pred_class, confidence = predict_image(uploaded_image_path)
            prediction = f"Class: {pred_class} (Confidence: {confidence * 100:.2f}%)"

    else:
        form = ImageUploadForm()

    return render(request, 'upload.html', {
        'form': form,
        'prediction': prediction,
        'original_image_url': original_image_url,
        'kirsch_image_url': kirsch_image_url,
    })
