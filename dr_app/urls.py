from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # example view (will add real one later)
    path('', views.upload_image, name='upload_image'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
