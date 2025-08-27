function getCurrentGW() {
  const url = "https://fantasy.premierleague.com/api/bootstrap-static/";
  const response = UrlFetchApp.fetch(url);
  const data = JSON.parse(response.getContentText());
  const currentEvent = data.events.find(event => event.is_current === true);
  if (currentEvent) {
    return currentEvent.id;
  } else {
    throw new Error("ไม่พบ Gameweek ปัจจุบัน");
  }
}

function updateFPLData() {
  try {
    const scriptProps = PropertiesService.getScriptProperties();
    const teamId = scriptProps.getProperty('teamId');
    if (!teamId) {
      throw new Error('❌ กรุณาตั้งค่า teamId ใน Script Properties ก่อน');
    }
    const currentGW = getCurrentGW();
    Logger.log(`Current GW: ${currentGW}`);
    updateReferenceData();
    updateTeamData(currentGW, teamId);
  } catch (e) {
    Logger.log(e);
    throw e;
  }
}

function updateReferenceData() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const bootstrapData = fetchBootstrap();
  updateReferenceSheets(ss, bootstrapData);
}

function updateTeamData(event, teamId) {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const fplSheet = ss.getSheetByName("FPL_Data") || ss.insertSheet("FPL_Data");
  fplSheet.clear();

  const bootstrapData = fetchBootstrap();
  const picksData = fetchEntryPicks(teamId, event);
  const historyData = fetchEntryHistory(teamId); // ใช้ history แทน event summary

  const result = mergeData(bootstrapData, picksData, historyData, event);

  fplSheet.appendRow(Object.keys(result[0]));
  result.forEach(row => {
    fplSheet.appendRow(Object.values(row));
  });
}


/**
 * 1) bootstrap-static → reference data
 */
function fetchBootstrap() {
  const url = "https://fantasy.premierleague.com/api/bootstrap-static/";
  const response = UrlFetchApp.fetch(url);
  return JSON.parse(response.getContentText());
}

function updateReferenceSheets(ss, bootstrapData) {
  writeToSheet(ss, "Players", bootstrapData.elements);
  writeToSheet(ss, "Teams", bootstrapData.teams);
  writeToSheet(ss, "ElementTypes", bootstrapData.element_types);
}

function writeToSheet(ss, sheetName, data) {
  let sheet = ss.getSheetByName(sheetName);
  if (!sheet) sheet = ss.insertSheet(sheetName);
  sheet.clear();
  if (data.length === 0) return;

  // เขียนหัวตาราง + ข้อมูล
  const headers = Object.keys(data[0]);
  sheet.appendRow(headers);
  data.forEach(obj => {
    sheet.appendRow(headers.map(h => obj[h]));
  });
}


/**
 * 2) entry/{teamId}/event/{event}/picks → squad picks
 */
function fetchEntryPicks(teamId, event) {
  const url = `https://fantasy.premierleague.com/api/entry/${teamId}/event/${event}/picks/`;
  const response = UrlFetchApp.fetch(url, { muteHttpExceptions: true });
  if (response.getResponseCode() !== 200) {
    throw new Error(`❌ fetchEntryPicks: Request failed for ${url} (HTTP ${response.getResponseCode()})\n${response.getContentText()}`);
  }
  return JSON.parse(response.getContentText());
}


/**
 * 3) entry/{teamId}/history → all GW summary
 */
function fetchEntryHistory(teamId) {
  const url = `https://fantasy.premierleague.com/api/entry/${teamId}/history/`;
  const response = UrlFetchApp.fetch(url, { muteHttpExceptions: true });
  if (response.getResponseCode() !== 200) {
    throw new Error(`❌ fetchEntryHistory: Request failed for ${url} (HTTP ${response.getResponseCode()})\n${response.getContentText()}`);
  }
  return JSON.parse(response.getContentText());
}

/**
 * รวมข้อมูล 3 API เข้าด้วยกัน
 * ใช้ historyData สำหรับคะแนน GW ปัจจุบัน
 */
function mergeData(bootstrapData, picksData, historyData, event) {
  const players = bootstrapData.elements;
  const teams = bootstrapData.teams;

  // หา GW ปัจจุบันใน history
  const gwHistory = historyData.current.find(gw => String(gw.event) === String(event));
  if (!gwHistory) {
    throw new Error(`❌ ไม่พบข้อมูลคะแนนสำหรับ GW ${event} ใน history`);
  }

  const merged = picksData.picks.map(pick => {
    const player = players.find(p => p.id === pick.element);
    const team = teams.find(t => t.id === player.team);

    return {
      player_id: player.id,
      player_name: player.web_name,
      team: team.name,
      position: player.element_type,
      is_captain: pick.is_captain,
      is_vice_captain: pick.is_vice_captain,
      multiplier: pick.multiplier,
      event_total: gwHistory.points,
      event_rank: gwHistory.rank,
      overall_points: gwHistory.total_points
    };
  });

  return merged;
}
