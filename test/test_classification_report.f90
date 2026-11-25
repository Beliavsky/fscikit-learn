program test_classification_report
  use sklearn_metrics,only:classification_report,classification_report_result
  use sklearn_common,only:dp,require
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use,intrinsic::ieee_arithmetic
  use,intrinsic::ieee_exceptions
  implicit none
  type(classification_report_result)::r,s
  type(csr_matrix)::a,b
  integer::y(4)=[0,0,1,1],p(4)=[0,1,1,1],ym(3,2),pm(3,2)
  logical::before(size(ieee_all)),after(size(ieee_all))
  call ieee_get_flag(ieee_all,before)
  r=classification_report(y,p)
  call ieee_get_flag(ieee_all,after)
  call require(all(before.eqv.after),'classification report preserves IEEE flags')
  call require(size(r%rows)==5.and.r%n_class_rows==2.and.r%micro_is_accuracy,'report includes accuracy and averages')
  call require(r%rows(1)%precision==1.and.r%rows(1)%recall==.5_dp,'analytic first class rates')
  call require(r%rows(3)%is_accuracy.and.r%rows(3)%precision==.75_dp,'analytic accuracy')
  call require(index(r%text,'accuracy')>0.and.index(r%text,'micro avg')==0,'accuracy text replacement')
  r=classification_report(y,p,labels=[1],output_dict=.true.)
  call require(.not.r%micro_is_accuracy.and.r%rows(2)%name=='micro avg'.and.len(r%text)==0,'subset has micro row')
  r=classification_report(y,p,labels=[0,1],target_names=['only'],output_dict=.true.)
  call require(r%n_class_rows==1.and.r%target_names_mismatch,'source zip truncation for explicit mismatched names')
  ym(1,:)=[1,0];ym(2,:)=[1,1];ym(3,:)=[0,1]
  pm(1,:)=[1,1];pm(2,:)=[0,1];pm(3,:)=[0,1]
  a=csr_from_dense(real(ym,dp));b=csr_from_dense(real(pm,dp))
  r=classification_report(ym,pm,sample_weight=[1._dp,2._dp,3._dp])
  s=classification_report(a,b,sample_weight=[1._dp,2._dp,3._dp])
  call require(r%text==s%text.and.size(r%rows)==6,'dense and CSR multilabel reports agree')
  call require(r%rows(6)%name=='samples avg','multilabel samples average')
  r=classification_report([2,4,1,4,3,0,1],[4,2,0,3,3,4,0],labels=[0], &
    sample_weight=[1._dp,-.5_dp,1._dp,-.5_dp,1._dp,-.5_dp,1._dp])
  call require(.not.ieee_is_negative(r%rows(2)%recall),'micro average includes NumPy positive-zero identity')
  call require(ieee_is_negative(r%rows(4)%precision),'weighted zero keeps negative denominator sign')
  print *,'Classification report rows, text, subset, names, CSR and IEEE state passed'
end program
