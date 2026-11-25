program reference_classification_report_invalid_driver
  use sklearn_common,only:dp
  use sklearn_metrics,only:classification_report,classification_report_result
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  implicit none
  type(classification_report_result)::r
  type(csr_matrix)::a,b
  integer::y(3)=[0,1,1],p(3)=[1,1,0],ym(3,2),pm(3,2)
  character(32)::mode
  call get_command_argument(1,mode)
  ym=1;pm=0
  select case(trim(mode))
  case('digits');r=classification_report(y,p,digits=-1)
  case('dict_digits');r=classification_report(y,p,digits=-1,output_dict=.true.)
  case('names');r=classification_report(y,p,target_names=['a'])
  case('length');r=classification_report(y,p(:2))
  case('shape');r=classification_report(ym,pm(:2,:))
  case('columns');r=classification_report(ym(:,:0),pm(:,:0))
  case('weight');r=classification_report(y,p,sample_weight=[1._dp])
  case('zero');r=classification_report(y,p,zero_division=2._dp)
  case('labels_low');r=classification_report(ym,pm,labels=[-1])
  case('labels_high');r=classification_report(ym,pm,labels=[2])
  case('indicator');ym(1,:)=[2,3];r=classification_report(ym,pm)
  case('csr_value')
    a=csr_from_dense(real(ym,dp));b=a;a%data(1)=2;r=classification_report(a,b)
  case('csr_duplicate')
    a=csr_from_dense(real(ym,dp));b=a;a%indices(2)=a%indices(1);r=classification_report(a,b)
  case('empty_labels');r=classification_report(y,p,labels=[integer::])
  case('empty_names');r=classification_report(y,p,labels=[0,1],target_names=[character(1)::])
  case default;error stop 'unknown report guard'
  end select
  error stop 'invalid classification report input accepted'
end program
