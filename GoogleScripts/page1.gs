function doPost(e) {
  let response;

  try {
    // Parse the incoming data
    let data = JSON.parse(e.postData.contents);

    // Define spreadsheet mapping (omitted for privacy reasons)
    const spreadsheetMapping = {
      /**
      Taken out for privacy reasons 
      */
    };

    // Get the spreadsheet URL for the provided age group
    const spreadsheetUrl = spreadsheetMapping[data.ageGroup];
    if (!spreadsheetUrl) {
      throw new Error("Invalid age group provided."); // Throw error if age group is invalid
    }

    // Open the spreadsheet by URL
    const sheets = SpreadsheetApp.openByUrl(spreadsheetUrl);

    // Define the suffix and create the sheet name
    const suffix = "TE"; 
    const sheetName = `Session${data.sessionNum}-${suffix}`;
    const sheet = sheets.getSheetByName(sheetName);
    if (!sheet) {
      throw new Error(`Sheet ${sheetName} not found.`); // Throw error if sheet is not found
    }

    // Append the data to the sheet
    sheet.appendRow([
      data.pinnieNumber,
      data.drill_1,
      data.drill_2,
      data.drill_3,
      data.drill_4,
      data.drill_5,
      data.drill_6,
      data.comments,
      data.average,
      data.evalNames,
      data.sessionNum,
      data.ageGroup
    ]);

    // Create a success response
    let responseObject = { result: 'success', data: data.pinnieNumber };
    let jsonString = JSON.stringify(responseObject);
    response = ContentService.createTextOutput(jsonString).setMimeType(ContentService.MimeType.JSON);
  } catch (error) {
    // Create an error response
    let errorObject = { result: 'error', message: error.message };
    let errorJsonString = JSON.stringify(errorObject);

    response = ContentService.createTextOutput(errorJsonString).setMimeType(ContentService.MimeType.JSON); 
  }

  // Return the response
  return response;
}
