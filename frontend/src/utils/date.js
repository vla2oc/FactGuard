
export const toISODate = (d) => {
  if (!(d instanceof Date)) d = new Date(d);
  return new Date(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()))
    .toISOString()
    .slice(0, 10);
};
export { toISODate as iso };
