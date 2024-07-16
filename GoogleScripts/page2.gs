function doPost(e) {
  let response;
  
  try {
    // Parse the incoming POST data
    let data = JSON.parse(e.postData.contents);

    // Mapping of age groups to their respective Google Sheets URLs
    const spreadsheetMapping = {
     /**
     Taken out for privacy reasons
     */
    };

    // Retrieve the URL of the spreadsheet for the given age group
    const spreadsheetUrl = spreadsheetMapping[data.ageGroup];
    if (!spreadsheetUrl) {
      throw new Error("Invalid age group provided.");
    }

    // Open the spreadsheet
    const sheets = SpreadsheetApp.openByUrl(spreadsheetUrl);
    const suffix = "SSE";
    const sheetName = `Session${data.sessionNum}-${suffix}`;
    const sheet = sheets.getSheetByName(sheetName);
    if (!sheet) {
      throw new Error(`Sheet ${sheetName} not found.`);
    }

    // Prepare a dictionary to store all player data
    let allPlayers = {};
    for (let i = 0; i < data.players.length; i++) {
      let roundNumber = i + 1;
      data.players[i].forEach(player => {
        if (!allPlayers[player.pinnieNumber]) {
          allPlayers[player.pinnieNumber] = {
            pinnieNumber: player.pinnieNumber,
            rounds: {
              "Round #1": "",
              "Round #2": "",
              "Round #3": "",
              "Round #4": "",
              "Round #5": "",
              "Round #6": "",
            },
            comments: []
          };
        }
        // Store field data and comments for each player and round
        allPlayers[player.pinnieNumber].rounds[`Round #${roundNumber}`] = data.field;
        allPlayers[player.pinnieNumber].comments.push(`Round ${roundNumber}: ${player.comments}`);
      });
    }

    // Retrieve existing data from the sheet
    const existingData = sheet.getDataRange().getValues();
    const header = existingData.shift(); // Remove the header row

    // Update or add new rows for each player
    Object.values(allPlayers).forEach(player => {
      let roundsData = player.rounds;
      let newComments = player.comments.join("; ");
      
      let newRow = [
        player.pinnieNumber,
        roundsData["Round #1"] || "",
        roundsData["Round #2"] || "",
        roundsData["Round #3"] || "",
        roundsData["Round #4"] || "",
        roundsData["Round #5"] || "",
        roundsData["Round #6"] || "",
        "", // Empty columns for future use
        "", // Empty columns for future use
        "", // Empty columns for future use
        newComments
      ];

      let playerExists = false;
      for (let i = 0; i < existingData.length; i++) {
        if (Number(existingData[i][0]) == Number(player.pinnieNumber)) {
          let currRow = existingData[i];
          // Merge existing data with new data
          for (let j = 1; j <= 6; j++) {
            if (!newRow[j] && currRow[j]) {
              newRow[j] = currRow[j];
            }
          }
          // Merge existing comments with new comments
          if (currRow[10]) {
            newRow[10] = currRow[10] + "; " + newComments;
          }
          // Update the existing row in the sheet
          sheet.getRange(i + 2, 1, 1, newRow.length).setValues([newRow]);
          playerExists = true;
          break;
        }
      }

      // Append a new row if the player doesn't already exist in the sheet
      if (!playerExists) {
        sheet.appendRow(newRow);
      }
    });

    // Prepare the success response
    let responseObject = { result: 'success', data: Object.keys(allPlayers) };
    let jsonString = JSON.stringify(responseObject);
    response = ContentService.createTextOutput(jsonString).setMimeType(ContentService.MimeType.JSON);
  } catch (error) {
    // Prepare the error response with debugging information
    let errorObject = {
      result: 'error',
      message: error.message,
      stack: error.stack,
      data: JSON.stringify(data),
      allPlayers: JSON.stringify(allPlayers)
    };
    let errorJsonString = JSON.stringify(errorObject);

    response = ContentService.createTextOutput(errorJsonString).setMimeType(ContentService.MimeType.JSON); 
  }

  return response;
}
