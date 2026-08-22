# What We Actually Did (plain-language explainer)

This file is for you, not the judges — so if someone asks "so what does your code do?", you have a simple answer ready. One section per level, added as we finish each one.

---

## Level 1 — Detection & Tracking

**The task in one sentence:** watch drone video of a road, draw a box around every car/truck/bus/motorbike/person, and make sure the same box keeps the same "name tag" as that vehicle moves through the video — even when it's briefly hidden behind another car or a tree.

### The two separate problems

There are actually **two different jobs** happening, and it's worth keeping them mentally separate:

1. **Detection** — "is there a car in this single frame, and where exactly?" This is a pure image problem. Show the AI one picture, it draws boxes.
2. **Tracking** — "is the car in *this* frame the same car as in the *previous* frame?" This is a video problem — it's about connecting the dots between frames over time so the same vehicle keeps the same ID number the whole way through, instead of getting a new random ID every frame.

Judges score you on getting both right, especially the "stable ID" part — a system that keeps relabeling the same truck as five different trucks isn't actually tracking anything.

### How detection works here: YOLOv8n

We used a model called **YOLOv8n** ("You Only Look Once", version 8, "n" for nano — the smallest, fastest size). It's a neural network that's already been trained by its creators on a huge public dataset of everyday photos (called COCO) containing millions of examples of cars, people, buses, etc. — so it already knows what these things look like before we ever touched it.

**We did not train our own model.** Two reasons: (1) the hackathon gave us zero labeled examples — no "here's a car, here's a truck" answer key to learn from — so there was nothing to train on even if we wanted to. (2) Training a model from scratch takes hours to days; we had about 7. So we used YOLOv8n exactly as it comes out of the box, pointed it at our drone footage, and trusted that "a car looks like a car" whether it's a street photo or a drone shot.

### How tracking works here: BoT-SORT

Once YOLOv8n finds boxes in every single frame, we need something to link "box #47 in frame 100" to "box #47 in frame 101" if it's the same real vehicle. That linking algorithm is called a **tracker**, and we used one called **BoT-SORT**.

The simpler, more common alternative is called ByteTrack — it basically assumes the camera itself is standing still and only the objects move, and it connects boxes across frames mostly by "this box is close to where the last one was." That assumption breaks on drone footage, because **the camera itself is constantly moving and panning** — so a stationary car's box also drifts across the frame just because the drone moved, and a simple tracker can get confused by that. BoT-SORT adds a step that specifically accounts for the camera's own motion first, so it doesn't mistake "the drone panned" for "everything on the ground moved." That's the whole reason we picked it over the simpler option.

### The one class the AI doesn't know: LGV vs HGV

The hackathon wants trucks split into two types: LGV (light goods vehicle — think delivery van, pickup) and HGV (heavy goods vehicle — think semi-truck, lorry). Problem: YOLOv8n's training data has one single "truck" bucket — it was never taught the light/heavy distinction, because that's not a category in the dataset it learned from.

Our fix is basically **estimating how physically big the truck is in real-world meters, then setting a size cutoff.** Here's the trick: we know how high up the drone was flying (this comes from the drone's own flight log, called telemetry) and roughly how wide a view its camera captures. With those two numbers plus simple geometry (the same math as "how big does a person look from a plane window"), we can convert "this truck's box is X pixels wide" into "this truck is approximately Y meters long in real life." If it's under about 7 meters, we call it an LGV; over that, an HGV. It's an estimate, not a certified measurement — but it's a real, defensible calculation, not a guess, and the 7-meter cutoff is something we can and will fine-tune once we've actually looked at footage.

If the flight-log data isn't available or doesn't parse cleanly, we fall back to a cruder version of the same idea: comparing the truck's box size to the size of ordinary cars visible in the very same frame — a truck box more than 2.5x the size of a typical car nearby gets called HGV, otherwise LGV.

### What the output actually is

For each video, our code produces one spreadsheet-style file (a CSV) with one row per detected object per frame, recording: which tracked vehicle/person it is (an ID number), which video frame, the timestamp, what class it is (car/LGV/HGV/bus/motorcycle/pedestrian), the box's pixel coordinates, and how confident the model was.

This file is deliberately kept simple and generic — not because L1 needs it that way, but because **Levels 2 through 5 all build on top of this exact same data.** Object-level insight (L2), counting/aggregating patterns (L3), and tying it to the map (L4) all just read more meaning out of the same rows rather than needing us to redo the detection work.

### How we checked it actually works (no answer key exists)

Since nobody gave us correct answers to grade against, we can't just compute "% accuracy." Instead we render a short clip of the video with the boxes and ID numbers drawn directly onto it, and watch it — do the boxes track real vehicles, do IDs stay stable through occlusion, do the LGV/HGV labels look sane for what's clearly a van vs. clearly a semi. That same clip also doubles as the demo video the submission requires, so the sanity check isn't throwaway work.

---

*(Sections for Levels 2-5 get added here as we build them.)*
