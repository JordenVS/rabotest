class Event:
    def __init__(self, eid, activity, objects, timestamp=None, object_types=None):
        self.eid = eid
        self.activity = activity
        self.objects = objects
        self.timestamp = timestamp
        self.object_types = object_types or set()