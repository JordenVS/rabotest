class Event:
    def __init__(self, eid, activity, objects, object_types=None):
        self.eid = eid
        self.activity = activity
        self.objects = objects
        self.object_types = object_types or set()