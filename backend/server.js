const express = require('express');
const fetch = require('node-fetch-commonjs');
const dotenv = require('dotenv');
const cors = require('cors');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5002;

app.use(cors());
app.use(express.json());

const TECHNICAL_EVAL_URL = "https://script.google.com/macros/s/AKfycbzaFHFNkz88WtKNigECnAbo1BpOs9SmMhoc7hr7XnqHYlZ-YkVPmp7sGt_EM6S4AmY/exec";
const SMALL_SIDED_EVAL_URL = "https://script.google.com/macros/s/AKfycbyCVPm71Wo9OaLiYgbvHhq3gJFwq1CutkzB5f0J0wKlS5pOT6MDKkQfUa2qHY5D7g/exec";
const FINAL_EVAL_URL = "https://script.google.com/macros/s/AKfycbxKRUDxVULNA-FUAGZzXNrYvh6iBMGulDpT9Mx0aDKHOKje2KF_oAWfiij_QkSjtfU/exec";

app.get('/', (req, res) => {
  res.send('Hello, world! Welcome to our project! This project was built by Arav Arora and Will Palaia!');
});

const processFinalPlayerEvaluations = (player, fieldNumber, sessionNum, ageGroup, index) => {
  const delay = 1000 * index;
  setTimeout(() => {
    fetch(FINAL_EVAL_URL, {
      method: "POST",
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        field: fieldNumber,
        pinnieNumber: player.pinnieNumber,
        technical: player.stat.technical,
        tactical: player.stat.tactical,
        effort: player.stat.effort,
        comments: player.comments,
        average: player.average,
        sessionNum: sessionNum,
        ageGroup: ageGroup,
        rank: player.rank,
      })
    })
    .then(res => res.json())
    .then(data => {
      if (data.result !== 'success') {
        throw new Error(`Error: ${data.message || 'Unknown error'}`);
      }
      console.log('Success:', data);
    })
    .catch(error => {
      console.error('Error:', error.message || error);
    });
  }, delay);
};

app.post('/submitFE', (req, res) => {
  const { players, fieldNumber, sessionNum, ageGroup } = req.body;

  players.forEach((player, index) => {
    processFinalPlayerEvaluations(player, fieldNumber, sessionNum, ageGroup, index);
  });

  res.status(200).json({ message: 'Data processing started' });
});

const processTechnicalPlayerEvaluations = (player, evalNames, sessionNum, ageGroup, index) => {
  const delay = 1000 * index;
  setTimeout(() => {
    fetch(TECHNICAL_EVAL_URL, {
      method: "POST",
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        pinnieNumber: player.pinnieNumber || 0,
        drill_1: player.drill["#1"],
        drill_2: player.drill["#2"],
        drill_3: player.drill["#3"],
        drill_4: player.drill["#4"],
        drill_5: player.drill["#5"],
        drill_6: player.drill["#6"],
        comments: player.comments || "",
        average: player.average,
        evalNames: evalNames,
        ageGroup: ageGroup,
        sessionNum: sessionNum
      })
    })
    .then(res => res.json())
    .then(data => {
      if (data.result !== 'success') {
        throw new Error(`Error: ${data.message || 'Unknown error'}`);
      }
      console.log('Success:', data);
    })
    .catch(error => {
      console.error('Error:', error.message || error);
    });
  }, delay);
};

app.post('/submitTE', (req, res) => {
  const { players, sessionNum, ageGroup, evalNames } = req.body;

  players.forEach((player, index) => {
    processTechnicalPlayerEvaluations(player, evalNames, sessionNum, ageGroup, index);
  });

  res.status(200).json({ message: 'Data processing started' });
});

const processSmallSidedEvaluations = (player, evalNames, sessionNum, ageGroup, index) => {
  const delay = 1000 * index;
  setTimeout(() => {
    fetch(SMALL_SIDED_EVAL_URL, {
      method: "POST",
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        pinnieNumber: player.pinnieNumber || 0,
        comments: player.comments || "",
        category: player.category,
        evalNames: evalNames,
        ageGroup: ageGroup,
        sessionNum: sessionNum
      })
    })
    .then(res => res.json())
    .then(data => {
      if (data.result !== 'success') {
        throw new Error(`Error: ${data.message || 'Unknown error'}`);
      }
      console.log('Success:', data);
    })
    .catch(error => {
      console.error('Error:', error.message || error);
    });
  }, delay);
};

app.post('/submitSSE', (req, res) => {
  const { players, sessionNum, ageGroup, evalNames } = req.body;

  players.forEach((player, index) => {
    processSmallSidedEvaluations(player, evalNames, sessionNum, ageGroup, index);
  });

  res.status(200).json({ message: 'Data processing started' });
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
