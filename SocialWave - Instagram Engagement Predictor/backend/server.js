const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const bodyParser = require("body-parser");
const bcrypt = require("bcryptjs");
const multer = require("multer");

const app = express();
app.use(cors());
app.use(bodyParser.json());

// === MongoDB Connection ===
mongoose.connect("mongodb+srv://dbcerin:dbcerin123@cluster0.ua84d.mongodb.net/socialwave_db")
    .then(() => console.log("✅ Connected to MongoDB Atlas"))
    .catch(err => console.error("❌ MongoDB Connection Error:", err));

// === User Schema with multiple posts ===
const userSchema = new mongoose.Schema({
    fullname: String,
    email: String,
    password: String,
    posts: [
        {
            image: {
                data: Buffer,
                contentType: String
            },
            caption: String,
            hashtags: String,
            engagementRate: Number,
            createdAt: {
                type: Date,
                default: Date.now
            }
        }
    ]
});


const User = mongoose.model("User", userSchema);

// === Signup Route ===
app.post("/signup", async (req, res) => {
    try {
        const { fullname, email, password, confirmPassword } = req.body;

        if (password !== confirmPassword) {
            return res.status(400).json({ message: "Passwords do not match!" });
        }

        const existingUser = await User.findOne({ email });
        if (existingUser) {
            return res.status(400).json({ message: "User already exists!" });
        }

        const hashedPassword = await bcrypt.hash(password, 10);
        const newUser = new User({ fullname, email, password: hashedPassword });
        await newUser.save();

        res.status(201).json({ message: "User registered successfully!" });
    } catch (error) {
        console.error("❌ Signup error:", error);
        res.status(500).json({ message: "Internal Server Error" });
    }
});

// === Login Route ===
app.post("/login", async (req, res) => {
    const { email, password } = req.body;

    try {
        const user = await User.findOne({ email });
        if (!user) {
            return res.status(400).json({ message: "User not found" });
        }

        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) {
            return res.status(400).json({ message: "Invalid credentials" });
        }

        res.json({ message: "Login successful", email });
    } catch (error) {
        console.error("❌ Login error:", error);
        res.status(500).json({ message: "Server error" });
    }
});

// === Upload Post ===
const storage = multer.memoryStorage();
const upload = multer({ storage });

// Upload Route
app.post("/upload", upload.single("image"), async (req, res) => {
    try {
        const { email, caption, hashtags, engagementRate } = req.body;

        const user = await User.findOne({ email });
        if (!user) {
            return res.status(404).json({ message: "User not found" });
        }

        const newPost = {
            image: {
                data: req.file.buffer,
                contentType: req.file.mimetype
            },
            caption,
            hashtags,
            engagementRate,
            createdAt: new Date()
        };

        user.posts.push(newPost);
        await user.save();

        res.status(200).json({ message: "Post data saved with engagement rate!" });
    } catch (error) {
        console.error("❌ Upload error:", error);
        res.status(500).json({ message: "Failed to upload and save post" });
    }
});


// === Get User Posts ===

app.get("/posts/:email", async (req, res) => {
    try {
        const { email } = req.params;
        const user = await User.findOne({ email });

        if (!user || !user.posts || user.posts.length === 0) {
            return res.status(404).json({ message: "No posts found for this user" });
        }

        // Format posts
        const postsSummary = user.posts.map(post => ({
            caption: post.caption,
            hashtags: post.hashtags,
            engagementRate: post.engagementRate,
            createdAt: post.createdAt,
            image: post.image?.data
                ? `data:${post.image.contentType};base64,${post.image.data.toString("base64")}`
                : null
        }));

        res.json({ posts: postsSummary });
    } catch (error) {
        console.error("❌ Fetch posts error:", error);
        res.status(500).json({ message: "Failed to retrieve posts" });
    }
});

app.get('/test', (req, res) => {
    res.send("Hello from test!");
  });
  

// === Start Server ===
app.listen(5000, () => {
    console.log("🚀 Server running on port 5000");
});
