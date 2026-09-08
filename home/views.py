from django.shortcuts import render


def index(request):
    students = [
        {
            "name": "Mohammed Haris Shaikh",
            "matriculation": "679516"
        }
    ]

    projects = [
        {"name": "Project 1", "url_name": "project1:index"},
        {"name": "Project 2", "url_name": "project2:index"},
        {"name": "Project 3", "url_name": "expert_interface"},
        {"name": "Project 4", "url_name": "project4:index"},
    ]

    return render(request, "home/index.html", {
        "students": students,
        "projects": projects
    })