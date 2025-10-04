import AxiosInstance from "./axios";
export const getEvents = async ({ start_date, end_date, event_type, sub_event_type }) => {
  try {
    const response = await AxiosInstance.get("/", {
      params: { start_date, end_date, event_type, sub_event_type },
    });
    return response.data;
  } catch (error) {
    console.error("Ошибка при получении событий:", error);
    return [];
  }
};
