program classification_report_example
  use sklearn,only:classification_report,classification_report_result,dp
  implicit none
  type(classification_report_result)::report
  report=classification_report([0,0,1,1,2],[0,1,1,1,0], &
    target_names=['class zero','class one ','class two '])
  print '(a)',report%text
  report=classification_report([0,0,1,1,2],[0,1,1,1,0],output_dict=.true., &
    sample_weight=[1._dp,1._dp,2._dp,1._dp,.5_dp])
  print *,'Owned accuracy value:',report%rows(report%n_class_rows+1)%precision
end program
