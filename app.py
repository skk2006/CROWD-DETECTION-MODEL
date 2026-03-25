import os
from dotenv import load_dotenv
load_dotenv()
import cloudinary
import cloudinary.uploader
from datetime import datetime
from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, jsonify)
from werkzeug.utils import secure_filename
from PIL import Image
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False
import config
import db
from detection import count_people

cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
    secure=True
)

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

# Global error handler — always return JSON for AJAX requests
@app.errorhandler(Exception)
def handle_exception(e):
    if request.accept_mimetypes.best == 'application/json' or request.is_json or request.path == '/upload':
        return jsonify({"success": False, "error": f"Internal server error: {str(e)}"}), 500
    return str(e), 500

@app.errorhandler(413)
def handle_too_large(e):
    return jsonify({"success": False, "error": "File too large."}), 413

# Temp folder for images during detection (deleted immediately after)
os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
  

# ────────────── Common User Routes ──────────────

@app.route("/")
def index():
    events = db.load_events()
    return render_template("upload.html", events=events)


@app.route("/upload", methods=["POST"])
def upload():
    events = db.load_events()
    place = request.form.get("place", "").strip()
    
    if not place:
        return jsonify({"success": False, "error": "Please enter a place name."}), 400
    
    # Add to events list if new place
    if place not in events:
        events.append(place)
        db.save_events(events)
    
    event = place

    file = request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"success": False, "error": "No image selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file type. Use JPG, PNG, BMP, WEBP, TIFF, or HEIC."}), 400

    # Save uploaded image temporarily for detection
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_filename = secure_filename(file.filename)
    ext = original_filename.rsplit(".", 1)[1].lower() if "." in original_filename else ""
    filepath = os.path.join(config.UPLOAD_FOLDER, f"{timestamp}_{original_filename}")
    file.save(filepath)

    # Detect HEIC/HEIF by file content (magic bytes) — iOS may not set extension
    is_heic = ext in ("heic", "heif")
    if not is_heic:
        try:
            with open(filepath, "rb") as fh:
                header = fh.read(12)
            # HEIF/HEIC files contain 'ftyp' at byte 4, followed by heic/heix/mif1/msf1
            if len(header) >= 12 and header[4:8] == b'ftyp':
                brand = header[8:12].decode('ascii', errors='ignore').lower()
                if brand in ('heic', 'heix', 'mif1', 'msf1', 'hevc'):
                    is_heic = True
        except Exception:
            pass

    # Convert HEIC/HEIF to JPEG so the detection model can process it
    if is_heic:
        if not HEIC_SUPPORTED:
            try:
                os.remove(filepath)
            except Exception:
                pass
            return jsonify({"success": False, "error": "HEIC support is not available on this server. Please convert the image to JPEG first."}), 400
        try:
            img = Image.open(filepath)
            jpeg_path = filepath.rsplit(".", 1)[0] + ".jpg"
            img.convert("RGB").save(jpeg_path, "JPEG", quality=95)
            os.remove(filepath)
            filepath = jpeg_path
        except Exception as e:
            try:
                os.remove(filepath)
            except Exception:
                pass
            return jsonify({"success": False, "error": f"Failed to convert HEIC image: {str(e)}"}), 500

    # Run detection — returns annotated JPEG bytes in memory
    try:
        head_count, annotated_bytes = count_people(filepath)
    except Exception as e:
        try:
            os.remove(filepath)
        except Exception:
            pass
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        # Always delete the temp original — it is no longer needed
        try:
            os.remove(filepath)
        except Exception:
            pass

    # Upload annotated image to Cloudinary (permanent cloud storage)
    annotated_url = None
    cloudinary_public_id = None
    cloud_upload_error = None
    try:
        result = cloudinary.uploader.upload(
            annotated_bytes,
            folder="crowd_detection",
            resource_type="image"
        )
        annotated_url = result.get("secure_url")
        cloudinary_public_id = result.get("public_id")
    except Exception as e:
        cloud_upload_error = str(e)

    # Save record to MongoDB (or local file) regardless of cloud upload status
    records = db.load_records()
    new_id = max((r["id"] for r in records), default=0) + 1
    records.append({
        "id": new_id,
        "event": event,
        "annotated_url": annotated_url,
        "cloudinary_public_id": cloudinary_public_id,
        "head_count": head_count,
        "timestamp": datetime.now().isoformat(),
        "cloud_upload_error": cloud_upload_error,
    })
    db.save_records(records)

    resp = {"success": True, "head_count": head_count, "event": event}
    if cloud_upload_error:
        resp["warning"] = "Cloud upload failed: " + cloud_upload_error
    return jsonify(resp)




# ────────────── Admin Routes ──────────────

def admin_required(f):
    """Decorator to protect admin routes."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated


@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    if session.get("admin_logged_in"):
        return redirect(url_for("admin_dashboard"))

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == config.ADMIN_USERNAME and password == config.ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Invalid credentials.", "error")

    return render_template("admin/login.html")


@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    records = db.load_records()
    events = db.load_events()

    # Analytics
    total_uploads = len(records)
    total_people = sum(r["head_count"] for r in records)
    avg_count = round(total_people / total_uploads, 1) if total_uploads > 0 else 0
    max_count = max((r["head_count"] for r in records), default=0)
    min_count = min((r["head_count"] for r in records), default=0)

    # Per-event analytics
    event_stats = {}
    for event in events:
        event_records = [r for r in records if r["event"] == event]
        count = len(event_records)
        total = sum(r["head_count"] for r in event_records)
        event_stats[event] = {
            "uploads": count,
            "total_people": total,
            "avg_count": round(total / count, 1) if count > 0 else 0,
            "max_count": max((r["head_count"] for r in event_records), default=0),
        }

    # Recent uploads (last 10)
    recent = list(reversed(records[-10:]))

    # Daily upload counts (for chart)
    daily_counts = {}
    for r in records:
        day = r["timestamp"][:10]
        daily_counts[day] = daily_counts.get(day, 0) + 1

    return render_template("admin/dashboard.html",
                           records=records, events=events,
                           total_uploads=total_uploads,
                           total_people=total_people,
                           avg_count=avg_count,
                           max_count=max_count,
                           min_count=min_count,
                           event_stats=event_stats,
                           recent=recent,
                           daily_counts=daily_counts)


@app.route("/admin/photos")
@admin_required
def admin_photos():
    records = db.load_records()
    events = db.load_events()
    selected_event = request.args.get("event", "all")

    if selected_event != "all":
        filtered = [r for r in records if r["event"] == selected_event]
    else:
        filtered = records

    return render_template("admin/photos.html",
                           records=list(reversed(filtered)),
                           events=events,
                           selected_event=selected_event)


@app.route("/admin/places")
@admin_required
def admin_places():
    events = db.load_events()
    records = db.load_records()

    # Count uploads per place
    place_counts = {}
    for event in events:
        place_counts[event] = len([r for r in records if r["event"] == event])

    return render_template("admin/places.html",
                           events=events, place_counts=place_counts)


@app.route("/admin/places/add", methods=["POST"])
@admin_required
def admin_add_place():
    name = request.form.get("name", "").strip()
    if not name:
        flash("Place name cannot be empty.", "error")
        return redirect(url_for("admin_places"))

    events = db.load_events()
    if name in events:
        flash("Place already exists.", "error")
        return redirect(url_for("admin_places"))

    events.append(name)
    db.save_events(events)
    flash(f"Place '{name}' added successfully.", "success")
    return redirect(url_for("admin_places"))


@app.route("/admin/places/delete/<int:index>", methods=["POST"])
@admin_required
def admin_delete_place(index):
    events = db.load_events()
    if 0 <= index < len(events):
        removed = events.pop(index)
        db.save_events(events)
        flash(f"Place '{removed}' deleted.", "success")
    return redirect(url_for("admin_places"))


@app.route("/admin/places/rename", methods=["POST"])
@admin_required
def admin_rename_place():
    old_name = request.form.get("old_name", "").strip()
    new_name = request.form.get("new_name", "").strip()

    if not new_name:
        flash("New name cannot be empty.", "error")
        return redirect(url_for("admin_places"))

    events = db.load_events()
    if old_name in events:
        idx = events.index(old_name)
        events[idx] = new_name
        db.save_events(events)

        # Rename in data records too
        records = db.load_records()
        for r in records:
            if r["event"] == old_name:
                r["event"] = new_name
        db.save_records(records)

        flash(f"Renamed '{old_name}' to '{new_name}'.", "success")
    return redirect(url_for("admin_places"))


@app.route("/admin/records/edit/<int:record_id>", methods=["GET", "POST"])
@admin_required
def admin_edit_record(record_id):
    records = db.load_records()
    events = db.load_events()
    record = None
    record_idx = None

    for i, r in enumerate(records):
        if r.get("id") == record_id:
            record = r
            record_idx = i
            break

    if record is None:
        flash("Record not found.", "error")
        return redirect(url_for("admin_dashboard"))

    if request.method == "POST":
        new_event = request.form.get("event")
        new_count = request.form.get("head_count")

        if new_event and new_event in events:
            records[record_idx]["event"] = new_event
        if new_count and new_count.isdigit():
            records[record_idx]["head_count"] = int(new_count)

        db.save_records(records)
        flash("Record updated successfully.", "success")
        return redirect(url_for("admin_dashboard"))

    return render_template("admin/edit_record.html",
                           record=record, events=events)


@app.route("/admin/records/delete/<int:record_id>", methods=["POST"])
@admin_required
def admin_delete_record(record_id):
    records = db.load_records()
    for r in records:
        if r.get("id") == record_id:
            try:
                cloudinary.uploader.destroy(r["cloudinary_public_id"])
            except Exception:
                pass
            break
    records = [r for r in records if r.get("id") != record_id]
    db.save_records(records)
    flash("Record deleted.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/photos/delete/<int:record_id>", methods=["POST"])
@admin_required
def admin_delete_photo(record_id):
    records = db.load_records()
    for r in records:
        if r.get("id") == record_id:
            try:
                cloudinary.uploader.destroy(r["cloudinary_public_id"])
            except Exception:
                pass
            break
    records = [r for r in records if r.get("id") != record_id]
    db.save_records(records)
    flash("Photo deleted.", "success")
    return redirect(url_for("admin_photos"))


@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("admin_login"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
