const express = require("express");
const axios = require("axios");
const cors = require("cors");

const app = express();
app.use(cors());

const PORT = 3000;

app.get("/api/get-therapists", async (req, res) => {
  const { lat, lng } = req.query;

  if (!lat || !lng) {
    return res.status(400).json({ error: "Latitude and longitude required" });
  }

  try {
    const response = await axios.get("https://nominatim.openstreetmap.org/search", {
      params: {
        q: "therapist",  // Search term
        format: "json",
        limit: 5,  // Get top 5 results
        lat,
        lon: lng
      },
      headers: {
        "User-Agent": "therapist-finder-app/1.0"
      }
    });

    // Filter out invalid results
    const results = response.data.map(place => ({
      name: place.display_name.split(",")[0],
      address: place.display_name,
      lat: place.lat,
      lon: place.lon
    }));

    if (results.length === 0) {
      return res.json({ message: "No nearby therapists found." });
    }

    res.json({ results });
  } catch (error) {
    console.error("Nominatim error:", error.message);
    res.status(500).json({ error: "Could not fetch therapist data" });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
