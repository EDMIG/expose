for in=1:300
  fn=sprintf('./maya/%4.4d.txt', in);
  txy=[tx(:,in) ty(:,in)];
  save(fn,'txy','-ascii');
end
