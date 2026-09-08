from django.contrib import admin
from django.urls import include, path
from django.views.generic import RedirectView
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    # Redirect root page to home
    path("", RedirectView.as_view(url="/home/", permanent=False)),

    path("home/", include("home.urls")),
    path("admin/", admin.site.urls),
    path("demos/", include("demos.urls")),
    path("project1/", include("project1.urls")),
    path("project2/", include("project2.urls")),
    path("project3/", include("project3.urls")),
    path("project4/", include("project4.urls")),
]

if settings.DEBUG:
    urlpatterns += static(
        settings.MEDIA_URL,
        document_root=settings.MEDIA_ROOT
    )