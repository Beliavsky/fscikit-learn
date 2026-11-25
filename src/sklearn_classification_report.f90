! Classification report composition: sklearn 1.6.1 metrics/_classification.py (BSD).
! Owned rows and text rendering are original MIT integration.
module sklearn_classification_report
  use sklearn_common,only:dp,require
  use sklearn_sparse,only:csr_matrix
  use sklearn_classification_metrics,only:precision_recall_fscore_support,report_labels=>sorted_classes
  use sklearn_resampling,only:float_text=>real_text
  use sklearn_ieee_math,only:ieee_difference
  use,intrinsic::iso_fortran_env,only:int64
  use,intrinsic::ieee_arithmetic
  use,intrinsic::ieee_exceptions
  implicit none
  private
  public::classification_report,classification_report_result,classification_report_row
  type::classification_report_row
    character(:),allocatable::name
    real(dp)::precision=0,recall=0,f1_score=0,support=0
    logical::is_accuracy=.false.
  end type
  type::classification_report_result
    type(classification_report_row),allocatable::rows(:)
    integer,allocatable::labels(:)
    integer::n_class_rows=0
    logical::micro_is_accuracy=.false.,target_names_mismatch=.false.
    character(:),allocatable::text
  end type
  interface classification_report
    module procedure report_vv
    module procedure report_bb
    module procedure report_ii
    module procedure report_ll
    module procedure report_ss
    module procedure report_si
    module procedure report_is
    module procedure report_sl
    module procedure report_ls
  end interface
contains
  function report_vv(y,p,labels,target_names,sample_weight,digits,output_dict,zero_division)result(report)
    integer,intent(in)::y(:),p(:)
    integer,optional,intent(in)::labels(:),digits
    character(*),optional,intent(in)::target_names(:)
    real(dp),optional,intent(in)::sample_weight(:),zero_division
    logical,optional,intent(in)::output_dict
    type(classification_report_result)::report
    real(dp),allocatable::pr(:),re(:),fs(:),su(:)
    integer,allocatable::chosen(:),observed(:)
    integer::j,a,nclass,names_count,ndigits
    logical::multi,dict,real_support,modes(size(ieee_all)),flags(size(ieee_all))
    real(dp)::total
    character(8),parameter::averages(4)=[character(8)::'micro','macro','weighted','samples']
    character(:),allocatable::heading
    call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
    if(present(digits))call require(digits>=0,'classification report digits must be nonnegative')
    call ieee_set_halting_mode(ieee_all,.false.)
    multi=.false.;observed=report_labels([y,p])
    chosen=observed;if(present(labels))chosen=labels
    report%labels=chosen;report%micro_is_accuracy=.not.multi
    if(.not.multi)then
      do j=1,size(observed)
        if(.not.any(chosen==observed(j)))report%micro_is_accuracy=.false.
      end do
    end if
    call precision_recall_fscore_support(y,p,pr,re,fs,su,labels=chosen, &
      sample_weight=sample_weight,zero_division=zero_division)
    real_support=present(sample_weight)
    if(.not.multi)real_support=real_support.or..not.any(y==p)
    nclass=size(chosen);names_count=nclass
    if(present(target_names))then
      report%target_names_mismatch=size(target_names)/=nclass
      if(.not.present(labels))call require(.not.report%target_names_mismatch, &
        'classification report target_names count differs from observed classes; specify labels')
      names_count=min(nclass,size(target_names))
    end if
    report%n_class_rows=names_count
    allocate(report%rows(names_count+3+merge(1,0,multi)))
    do j=1,names_count
      heading=integer_text(chosen(j))
      if(present(target_names))heading=trim(target_names(j))
      report%rows(j)=classification_report_row(heading,pr(j),re(j),fs(j),su(j),.false.)
    end do
    total=support_sum(su)
    do a=1,3+merge(1,0,multi)
      call precision_recall_fscore_support(y,p,pr,re,fs,su,labels=chosen,average=trim(averages(a)), &
        sample_weight=sample_weight,zero_division=zero_division)
      heading=trim(averages(a))//' avg'
      if(a==1.and.report%micro_is_accuracy)heading='accuracy'
      report%rows(names_count+a)=classification_report_row(heading,pr(1),re(1),fs(1),total, &
        a==1.and.report%micro_is_accuracy)
    end do
    dict=.false.;if(present(output_dict))dict=output_dict
    report%text=''
    if(.not.dict)then
      ndigits=2;if(present(digits))ndigits=digits
      call require(ndigits>=0,'classification report digits must be nonnegative for text')
      call render_report(report,ndigits,real_support,target_names)
    end if
    call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
  end function
  function report_bb(y,p,labels,target_names,sample_weight,digits,output_dict,zero_division)result(report)
    logical,intent(in)::y(:),p(:)
    integer,optional,intent(in)::labels(:),digits
    character(*),optional,intent(in)::target_names(:)
    real(dp),optional,intent(in)::sample_weight(:),zero_division
    logical,optional,intent(in)::output_dict
    type(classification_report_result)::report
    real(dp),allocatable::pr(:),re(:),fs(:),su(:)
    integer,allocatable::chosen(:),observed(:)
    integer::j,a,nclass,names_count,ndigits
    logical::multi,dict,real_support,modes(size(ieee_all)),flags(size(ieee_all))
    real(dp)::total
    character(8),parameter::averages(4)=[character(8)::'micro','macro','weighted','samples']
    character(:),allocatable::heading
    call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
    if(present(digits))call require(digits>=0,'classification report digits must be nonnegative')
    call ieee_set_halting_mode(ieee_all,.false.)
    multi=.false.;observed=report_labels(merge(1,0,[y,p]))
    chosen=observed;if(present(labels))chosen=labels
    report%labels=chosen;report%micro_is_accuracy=.not.multi
    if(.not.multi)then
      do j=1,size(observed)
        if(.not.any(chosen==observed(j)))report%micro_is_accuracy=.false.
      end do
    end if
    call precision_recall_fscore_support(y,p,pr,re,fs,su,labels=chosen, &
      sample_weight=sample_weight,zero_division=zero_division)
    real_support=present(sample_weight)
    if(.not.multi)real_support=real_support.or..not.any(y.eqv.p)
    nclass=size(chosen);names_count=nclass
    if(present(target_names))then
      report%target_names_mismatch=size(target_names)/=nclass
      if(.not.present(labels))call require(.not.report%target_names_mismatch, &
        'classification report target_names count differs from observed classes; specify labels')
      names_count=min(nclass,size(target_names))
    end if
    report%n_class_rows=names_count
    allocate(report%rows(names_count+3+merge(1,0,multi)))
    do j=1,names_count
      heading=integer_text(chosen(j))
      if(.not.multi.and..not.present(labels))then
        heading='False';if(chosen(j)/=0)heading='True'
      end if
      if(present(target_names))heading=trim(target_names(j))
      report%rows(j)=classification_report_row(heading,pr(j),re(j),fs(j),su(j),.false.)
    end do
    total=support_sum(su)
    do a=1,3+merge(1,0,multi)
      call precision_recall_fscore_support(y,p,pr,re,fs,su,labels=chosen,average=trim(averages(a)), &
        sample_weight=sample_weight,zero_division=zero_division)
      heading=trim(averages(a))//' avg'
      if(a==1.and.report%micro_is_accuracy)heading='accuracy'
      report%rows(names_count+a)=classification_report_row(heading,pr(1),re(1),fs(1),total, &
        a==1.and.report%micro_is_accuracy)
    end do
    dict=.false.;if(present(output_dict))dict=output_dict
    report%text=''
    if(.not.dict)then
      ndigits=2;if(present(digits))ndigits=digits
      call require(ndigits>=0,'classification report digits must be nonnegative for text')
      call render_report(report,ndigits,real_support,target_names)
    end if
    call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
  end function
  function report_ii(y,p,labels,target_names,sample_weight,digits,output_dict,zero_division)result(report)
    integer,intent(in)::y(:,:),p(:,:)
    integer,optional,intent(in)::labels(:),digits
    character(*),optional,intent(in)::target_names(:)
    real(dp),optional,intent(in)::sample_weight(:),zero_division
    logical,optional,intent(in)::output_dict
    type(classification_report_result)::report
    real(dp),allocatable::pr(:),re(:),fs(:),su(:)
    integer,allocatable::chosen(:),observed(:)
    integer::j,a,nclass,names_count,ndigits
    logical::multi,dict,real_support,modes(size(ieee_all)),flags(size(ieee_all))
    real(dp)::total
    character(8),parameter::averages(4)=[character(8)::'micro','macro','weighted','samples']
    character(:),allocatable::heading
    call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
    if(present(digits))call require(digits>=0,'classification report digits must be nonnegative')
    call ieee_set_halting_mode(ieee_all,.false.)
    call require(all(shape(y)==shape(p)),'classification report target/prediction shapes differ')
    call require(size(y,2)>0,'classification report matrices need columns')
    multi=size(y,2)>1
    if(multi)then
      observed=[(j-1,j=1,size(y,2))]
    else
      observed=report_labels([y(:,1),p(:,1)])
    end if
    chosen=observed;if(present(labels))chosen=labels
    report%labels=chosen;report%micro_is_accuracy=.not.multi
    if(.not.multi)then
      do j=1,size(observed)
        if(.not.any(chosen==observed(j)))report%micro_is_accuracy=.false.
      end do
    end if
    call precision_recall_fscore_support(y,p,pr,re,fs,su,labels=chosen, &
      sample_weight=sample_weight,zero_division=zero_division)
    real_support=present(sample_weight)
    if(.not.multi)real_support=real_support.or..not.any(y==p)
    nclass=size(chosen);names_count=nclass
    if(present(target_names))then
      report%target_names_mismatch=size(target_names)/=nclass
      if(.not.present(labels))call require(.not.report%target_names_mismatch, &
        'classification report target_names count differs from observed classes; specify labels')
      names_count=min(nclass,size(target_names))
    end if
    report%n_class_rows=names_count
    allocate(report%rows(names_count+3+merge(1,0,multi)))
    do j=1,names_count
      heading=integer_text(chosen(j))
      if(present(target_names))heading=trim(target_names(j))
      report%rows(j)=classification_report_row(heading,pr(j),re(j),fs(j),su(j),.false.)
    end do
    total=support_sum(su)
    do a=1,3+merge(1,0,multi)
      call precision_recall_fscore_support(y,p,pr,re,fs,su,labels=chosen,average=trim(averages(a)), &
        sample_weight=sample_weight,zero_division=zero_division)
      heading=trim(averages(a))//' avg'
      if(a==1.and.report%micro_is_accuracy)heading='accuracy'
      report%rows(names_count+a)=classification_report_row(heading,pr(1),re(1),fs(1),total, &
        a==1.and.report%micro_is_accuracy)
    end do
    dict=.false.;if(present(output_dict))dict=output_dict
    report%text=''
    if(.not.dict)then
      ndigits=2;if(present(digits))ndigits=digits
      call require(ndigits>=0,'classification report digits must be nonnegative for text')
      call render_report(report,ndigits,real_support,target_names)
    end if
    call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
  end function
  function report_ll(y,p,labels,target_names,sample_weight,digits,output_dict,zero_division)result(report)
    logical,intent(in)::y(:,:),p(:,:)
    integer,optional,intent(in)::labels(:),digits
    character(*),optional,intent(in)::target_names(:)
    real(dp),optional,intent(in)::sample_weight(:),zero_division
    logical,optional,intent(in)::output_dict
    type(classification_report_result)::report
    real(dp),allocatable::pr(:),re(:),fs(:),su(:)
    integer,allocatable::chosen(:),observed(:)
    integer::j,a,nclass,names_count,ndigits
    logical::multi,dict,real_support,modes(size(ieee_all)),flags(size(ieee_all))
    real(dp)::total
    character(8),parameter::averages(4)=[character(8)::'micro','macro','weighted','samples']
    character(:),allocatable::heading
    call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
    if(present(digits))call require(digits>=0,'classification report digits must be nonnegative')
    call ieee_set_halting_mode(ieee_all,.false.)
    call require(all(shape(y)==shape(p)),'classification report target/prediction shapes differ')
    call require(size(y,2)>0,'classification report matrices need columns')
    multi=size(y,2)>1
    if(multi)then
      observed=[(j-1,j=1,size(y,2))]
    else
      observed=report_labels(merge(1,0,[y(:,1),p(:,1)]))
    end if
    chosen=observed;if(present(labels))chosen=labels
    report%labels=chosen;report%micro_is_accuracy=.not.multi
    if(.not.multi)then
      do j=1,size(observed)
        if(.not.any(chosen==observed(j)))report%micro_is_accuracy=.false.
      end do
    end if
    call precision_recall_fscore_support(y,p,pr,re,fs,su,labels=chosen, &
      sample_weight=sample_weight,zero_division=zero_division)
    real_support=present(sample_weight)
    if(.not.multi)real_support=real_support.or..not.any(y.eqv.p)
    nclass=size(chosen);names_count=nclass
    if(present(target_names))then
      report%target_names_mismatch=size(target_names)/=nclass
      if(.not.present(labels))call require(.not.report%target_names_mismatch, &
        'classification report target_names count differs from observed classes; specify labels')
      names_count=min(nclass,size(target_names))
    end if
    report%n_class_rows=names_count
    allocate(report%rows(names_count+3+merge(1,0,multi)))
    do j=1,names_count
      heading=integer_text(chosen(j))
      if(.not.multi.and..not.present(labels))then
        heading='False';if(chosen(j)/=0)heading='True'
      end if
      if(present(target_names))heading=trim(target_names(j))
      report%rows(j)=classification_report_row(heading,pr(j),re(j),fs(j),su(j),.false.)
    end do
    total=support_sum(su)
    do a=1,3+merge(1,0,multi)
      call precision_recall_fscore_support(y,p,pr,re,fs,su,labels=chosen,average=trim(averages(a)), &
        sample_weight=sample_weight,zero_division=zero_division)
      heading=trim(averages(a))//' avg'
      if(a==1.and.report%micro_is_accuracy)heading='accuracy'
      report%rows(names_count+a)=classification_report_row(heading,pr(1),re(1),fs(1),total, &
        a==1.and.report%micro_is_accuracy)
    end do
    dict=.false.;if(present(output_dict))dict=output_dict
    report%text=''
    if(.not.dict)then
      ndigits=2;if(present(digits))ndigits=digits
      call require(ndigits>=0,'classification report digits must be nonnegative for text')
      call render_report(report,ndigits,real_support,target_names)
    end if
    call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
  end function
  function report_ss(y,p,labels,target_names,sample_weight,digits,output_dict,zero_division)result(report)
    type(csr_matrix),intent(in)::y,p
    integer,optional,intent(in)::labels(:),digits
    character(*),optional,intent(in)::target_names(:)
    real(dp),optional,intent(in)::sample_weight(:),zero_division
    logical,optional,intent(in)::output_dict
    type(classification_report_result)::report
    real(dp),allocatable::pr(:),re(:),fs(:),su(:)
    integer,allocatable::chosen(:),observed(:)
    integer::j,a,nclass,names_count,ndigits
    logical::multi,dict,real_support,modes(size(ieee_all)),flags(size(ieee_all))
    real(dp)::total
    character(8),parameter::averages(4)=[character(8)::'micro','macro','weighted','samples']
    character(:),allocatable::heading
    call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
    if(present(digits))call require(digits>=0,'classification report digits must be nonnegative')
    call ieee_set_halting_mode(ieee_all,.false.)
    multi=.true.;observed=[(j-1,j=1,y%n_cols)]
    chosen=observed;if(present(labels))chosen=labels
    report%labels=chosen;report%micro_is_accuracy=.not.multi
    if(.not.multi)then
      do j=1,size(observed)
        if(.not.any(chosen==observed(j)))report%micro_is_accuracy=.false.
      end do
    end if
    call precision_recall_fscore_support(y,p,pr,re,fs,su,labels=chosen, &
      sample_weight=sample_weight,zero_division=zero_division)
    real_support=present(sample_weight)
    nclass=size(chosen);names_count=nclass
    if(present(target_names))then
      report%target_names_mismatch=size(target_names)/=nclass
      if(.not.present(labels))call require(.not.report%target_names_mismatch, &
        'classification report target_names count differs from observed classes; specify labels')
      names_count=min(nclass,size(target_names))
    end if
    report%n_class_rows=names_count
    allocate(report%rows(names_count+3+merge(1,0,multi)))
    do j=1,names_count
      heading=integer_text(chosen(j))
      if(present(target_names))heading=trim(target_names(j))
      report%rows(j)=classification_report_row(heading,pr(j),re(j),fs(j),su(j),.false.)
    end do
    total=support_sum(su)
    do a=1,3+merge(1,0,multi)
      call precision_recall_fscore_support(y,p,pr,re,fs,su,labels=chosen,average=trim(averages(a)), &
        sample_weight=sample_weight,zero_division=zero_division)
      heading=trim(averages(a))//' avg'
      if(a==1.and.report%micro_is_accuracy)heading='accuracy'
      report%rows(names_count+a)=classification_report_row(heading,pr(1),re(1),fs(1),total, &
        a==1.and.report%micro_is_accuracy)
    end do
    dict=.false.;if(present(output_dict))dict=output_dict
    report%text=''
    if(.not.dict)then
      ndigits=2;if(present(digits))ndigits=digits
      call require(ndigits>=0,'classification report digits must be nonnegative for text')
      call render_report(report,ndigits,real_support,target_names)
    end if
    call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
  end function
  function report_si(y,p,labels,target_names,sample_weight,digits,output_dict,zero_division)result(report)
    type(csr_matrix),intent(in)::y
    integer,intent(in)::p(:,:)
    integer,optional,intent(in)::labels(:),digits
    character(*),optional,intent(in)::target_names(:)
    real(dp),optional,intent(in)::sample_weight(:),zero_division
    logical,optional,intent(in)::output_dict
    type(classification_report_result)::report
    real(dp),allocatable::pr(:),re(:),fs(:),su(:)
    integer,allocatable::chosen(:),observed(:)
    integer::j,a,nclass,names_count,ndigits
    logical::multi,dict,real_support,modes(size(ieee_all)),flags(size(ieee_all))
    real(dp)::total
    character(8),parameter::averages(4)=[character(8)::'micro','macro','weighted','samples']
    character(:),allocatable::heading
    call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
    if(present(digits))call require(digits>=0,'classification report digits must be nonnegative')
    call ieee_set_halting_mode(ieee_all,.false.)
    multi=.true.;observed=[(j-1,j=1,y%n_cols)]
    chosen=observed;if(present(labels))chosen=labels
    report%labels=chosen;report%micro_is_accuracy=.not.multi
    if(.not.multi)then
      do j=1,size(observed)
        if(.not.any(chosen==observed(j)))report%micro_is_accuracy=.false.
      end do
    end if
    call precision_recall_fscore_support(y,p,pr,re,fs,su,labels=chosen, &
      sample_weight=sample_weight,zero_division=zero_division)
    real_support=present(sample_weight)
    nclass=size(chosen);names_count=nclass
    if(present(target_names))then
      report%target_names_mismatch=size(target_names)/=nclass
      if(.not.present(labels))call require(.not.report%target_names_mismatch, &
        'classification report target_names count differs from observed classes; specify labels')
      names_count=min(nclass,size(target_names))
    end if
    report%n_class_rows=names_count
    allocate(report%rows(names_count+3+merge(1,0,multi)))
    do j=1,names_count
      heading=integer_text(chosen(j))
      if(present(target_names))heading=trim(target_names(j))
      report%rows(j)=classification_report_row(heading,pr(j),re(j),fs(j),su(j),.false.)
    end do
    total=support_sum(su)
    do a=1,3+merge(1,0,multi)
      call precision_recall_fscore_support(y,p,pr,re,fs,su,labels=chosen,average=trim(averages(a)), &
        sample_weight=sample_weight,zero_division=zero_division)
      heading=trim(averages(a))//' avg'
      if(a==1.and.report%micro_is_accuracy)heading='accuracy'
      report%rows(names_count+a)=classification_report_row(heading,pr(1),re(1),fs(1),total, &
        a==1.and.report%micro_is_accuracy)
    end do
    dict=.false.;if(present(output_dict))dict=output_dict
    report%text=''
    if(.not.dict)then
      ndigits=2;if(present(digits))ndigits=digits
      call require(ndigits>=0,'classification report digits must be nonnegative for text')
      call render_report(report,ndigits,real_support,target_names)
    end if
    call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
  end function
  function report_is(y,p,labels,target_names,sample_weight,digits,output_dict,zero_division)result(report)
    type(csr_matrix),intent(in)::p
    integer,intent(in)::y(:,:)
    integer,optional,intent(in)::labels(:),digits
    character(*),optional,intent(in)::target_names(:)
    real(dp),optional,intent(in)::sample_weight(:),zero_division
    logical,optional,intent(in)::output_dict
    type(classification_report_result)::report
    real(dp),allocatable::pr(:),re(:),fs(:),su(:)
    integer,allocatable::chosen(:),observed(:)
    integer::j,a,nclass,names_count,ndigits
    logical::multi,dict,real_support,modes(size(ieee_all)),flags(size(ieee_all))
    real(dp)::total
    character(8),parameter::averages(4)=[character(8)::'micro','macro','weighted','samples']
    character(:),allocatable::heading
    call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
    if(present(digits))call require(digits>=0,'classification report digits must be nonnegative')
    call ieee_set_halting_mode(ieee_all,.false.)
    multi=.true.;observed=[(j-1,j=1,size(y,2))]
    chosen=observed;if(present(labels))chosen=labels
    report%labels=chosen;report%micro_is_accuracy=.not.multi
    if(.not.multi)then
      do j=1,size(observed)
        if(.not.any(chosen==observed(j)))report%micro_is_accuracy=.false.
      end do
    end if
    call precision_recall_fscore_support(y,p,pr,re,fs,su,labels=chosen, &
      sample_weight=sample_weight,zero_division=zero_division)
    real_support=present(sample_weight)
    nclass=size(chosen);names_count=nclass
    if(present(target_names))then
      report%target_names_mismatch=size(target_names)/=nclass
      if(.not.present(labels))call require(.not.report%target_names_mismatch, &
        'classification report target_names count differs from observed classes; specify labels')
      names_count=min(nclass,size(target_names))
    end if
    report%n_class_rows=names_count
    allocate(report%rows(names_count+3+merge(1,0,multi)))
    do j=1,names_count
      heading=integer_text(chosen(j))
      if(present(target_names))heading=trim(target_names(j))
      report%rows(j)=classification_report_row(heading,pr(j),re(j),fs(j),su(j),.false.)
    end do
    total=support_sum(su)
    do a=1,3+merge(1,0,multi)
      call precision_recall_fscore_support(y,p,pr,re,fs,su,labels=chosen,average=trim(averages(a)), &
        sample_weight=sample_weight,zero_division=zero_division)
      heading=trim(averages(a))//' avg'
      if(a==1.and.report%micro_is_accuracy)heading='accuracy'
      report%rows(names_count+a)=classification_report_row(heading,pr(1),re(1),fs(1),total, &
        a==1.and.report%micro_is_accuracy)
    end do
    dict=.false.;if(present(output_dict))dict=output_dict
    report%text=''
    if(.not.dict)then
      ndigits=2;if(present(digits))ndigits=digits
      call require(ndigits>=0,'classification report digits must be nonnegative for text')
      call render_report(report,ndigits,real_support,target_names)
    end if
    call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
  end function
  function report_sl(y,p,labels,target_names,sample_weight,digits,output_dict,zero_division)result(report)
    type(csr_matrix),intent(in)::y
    logical,intent(in)::p(:,:)
    integer,optional,intent(in)::labels(:),digits
    character(*),optional,intent(in)::target_names(:)
    real(dp),optional,intent(in)::sample_weight(:),zero_division
    logical,optional,intent(in)::output_dict
    type(classification_report_result)::report
    real(dp),allocatable::pr(:),re(:),fs(:),su(:)
    integer,allocatable::chosen(:),observed(:)
    integer::j,a,nclass,names_count,ndigits
    logical::multi,dict,real_support,modes(size(ieee_all)),flags(size(ieee_all))
    real(dp)::total
    character(8),parameter::averages(4)=[character(8)::'micro','macro','weighted','samples']
    character(:),allocatable::heading
    call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
    if(present(digits))call require(digits>=0,'classification report digits must be nonnegative')
    call ieee_set_halting_mode(ieee_all,.false.)
    multi=.true.;observed=[(j-1,j=1,y%n_cols)]
    chosen=observed;if(present(labels))chosen=labels
    report%labels=chosen;report%micro_is_accuracy=.not.multi
    if(.not.multi)then
      do j=1,size(observed)
        if(.not.any(chosen==observed(j)))report%micro_is_accuracy=.false.
      end do
    end if
    call precision_recall_fscore_support(y,p,pr,re,fs,su,labels=chosen, &
      sample_weight=sample_weight,zero_division=zero_division)
    real_support=present(sample_weight)
    nclass=size(chosen);names_count=nclass
    if(present(target_names))then
      report%target_names_mismatch=size(target_names)/=nclass
      if(.not.present(labels))call require(.not.report%target_names_mismatch, &
        'classification report target_names count differs from observed classes; specify labels')
      names_count=min(nclass,size(target_names))
    end if
    report%n_class_rows=names_count
    allocate(report%rows(names_count+3+merge(1,0,multi)))
    do j=1,names_count
      heading=integer_text(chosen(j))
      if(present(target_names))heading=trim(target_names(j))
      report%rows(j)=classification_report_row(heading,pr(j),re(j),fs(j),su(j),.false.)
    end do
    total=support_sum(su)
    do a=1,3+merge(1,0,multi)
      call precision_recall_fscore_support(y,p,pr,re,fs,su,labels=chosen,average=trim(averages(a)), &
        sample_weight=sample_weight,zero_division=zero_division)
      heading=trim(averages(a))//' avg'
      if(a==1.and.report%micro_is_accuracy)heading='accuracy'
      report%rows(names_count+a)=classification_report_row(heading,pr(1),re(1),fs(1),total, &
        a==1.and.report%micro_is_accuracy)
    end do
    dict=.false.;if(present(output_dict))dict=output_dict
    report%text=''
    if(.not.dict)then
      ndigits=2;if(present(digits))ndigits=digits
      call require(ndigits>=0,'classification report digits must be nonnegative for text')
      call render_report(report,ndigits,real_support,target_names)
    end if
    call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
  end function
  function report_ls(y,p,labels,target_names,sample_weight,digits,output_dict,zero_division)result(report)
    type(csr_matrix),intent(in)::p
    logical,intent(in)::y(:,:)
    integer,optional,intent(in)::labels(:),digits
    character(*),optional,intent(in)::target_names(:)
    real(dp),optional,intent(in)::sample_weight(:),zero_division
    logical,optional,intent(in)::output_dict
    type(classification_report_result)::report
    real(dp),allocatable::pr(:),re(:),fs(:),su(:)
    integer,allocatable::chosen(:),observed(:)
    integer::j,a,nclass,names_count,ndigits
    logical::multi,dict,real_support,modes(size(ieee_all)),flags(size(ieee_all))
    real(dp)::total
    character(8),parameter::averages(4)=[character(8)::'micro','macro','weighted','samples']
    character(:),allocatable::heading
    call ieee_get_halting_mode(ieee_all,modes);call ieee_get_flag(ieee_all,flags)
    if(present(digits))call require(digits>=0,'classification report digits must be nonnegative')
    call ieee_set_halting_mode(ieee_all,.false.)
    multi=.true.;observed=[(j-1,j=1,size(y,2))]
    chosen=observed;if(present(labels))chosen=labels
    report%labels=chosen;report%micro_is_accuracy=.not.multi
    if(.not.multi)then
      do j=1,size(observed)
        if(.not.any(chosen==observed(j)))report%micro_is_accuracy=.false.
      end do
    end if
    call precision_recall_fscore_support(y,p,pr,re,fs,su,labels=chosen, &
      sample_weight=sample_weight,zero_division=zero_division)
    real_support=present(sample_weight)
    nclass=size(chosen);names_count=nclass
    if(present(target_names))then
      report%target_names_mismatch=size(target_names)/=nclass
      if(.not.present(labels))call require(.not.report%target_names_mismatch, &
        'classification report target_names count differs from observed classes; specify labels')
      names_count=min(nclass,size(target_names))
    end if
    report%n_class_rows=names_count
    allocate(report%rows(names_count+3+merge(1,0,multi)))
    do j=1,names_count
      heading=integer_text(chosen(j))
      if(present(target_names))heading=trim(target_names(j))
      report%rows(j)=classification_report_row(heading,pr(j),re(j),fs(j),su(j),.false.)
    end do
    total=support_sum(su)
    do a=1,3+merge(1,0,multi)
      call precision_recall_fscore_support(y,p,pr,re,fs,su,labels=chosen,average=trim(averages(a)), &
        sample_weight=sample_weight,zero_division=zero_division)
      heading=trim(averages(a))//' avg'
      if(a==1.and.report%micro_is_accuracy)heading='accuracy'
      report%rows(names_count+a)=classification_report_row(heading,pr(1),re(1),fs(1),total, &
        a==1.and.report%micro_is_accuracy)
    end do
    dict=.false.;if(present(output_dict))dict=output_dict
    report%text=''
    if(.not.dict)then
      ndigits=2;if(present(digits))ndigits=digits
      call require(ndigits>=0,'classification report digits must be nonnegative for text')
      call render_report(report,ndigits,real_support,target_names)
    end if
    call ieee_set_flag(ieee_all,flags);call ieee_set_halting_mode(ieee_all,modes)
  end function
  function integer_text(value)result(text)
    integer,intent(in)::value
    character(:),allocatable::text
    character(32)::buffer
    write(buffer,'(i0)')value;text=trim(buffer)
  end function
  recursive real(dp)function support_sum(a)result(total)
    ! Same NumPy pairwise ordering as the classification reductions.
    real(dp),intent(in)::a(:)
    real(dp)::v(8),left,right
    integer::n,i,j,whole,split
    n=size(a);total=-0._dp
    if(n==0)then
      total=0;return
    end if
    if(n<8)then
      do i=1,n
        total=ieee_difference(total,-a(i))
      end do
    else if(n<=128)then
      v=a(:8);whole=n-mod(n,8)
      do i=9,whole,8
        do j=1,8
          v(j)=ieee_difference(v(j),-a(i+j-1))
        end do
      end do
      left=ieee_difference(ieee_difference(v(1),-v(2)),-ieee_difference(v(3),-v(4)))
      right=ieee_difference(ieee_difference(v(5),-v(6)),-ieee_difference(v(7),-v(8)))
      total=ieee_difference(left,-right)
      do i=whole+1,n
        total=ieee_difference(total,-a(i))
      end do
    else
      split=n/2;split=split-mod(split,8)
      total=ieee_difference(support_sum(a(:split)),-support_sum(a(split+1:)))
    end if
  end function
  function padded(text,width)result(value)
    character(*),intent(in)::text
    integer,intent(in)::width
    character(:),allocatable::value
    value=repeat(' ',max(0,width-len(text)))//text
  end function
  function fixed(value,digits)result(text)
    real(dp),intent(in)::value
    integer,intent(in)::digits
    character(:),allocatable::text,buffer
    character(64)::fmt
    if(ieee_is_nan(value))then
      text='nan'
    else if(.not.ieee_is_finite(value))then
      text='inf';if(value<0)text='-inf'
    else
      ! A binary64 integral part needs at most 309 decimal digits.
      call require(digits<=huge(1)-320,'classification report text allocation overflow')
      allocate(character(digits+320)::buffer)
      write(fmt,'(a,i0,a,i0,a)')'(rn,f',digits+320,'.',digits,')'
      write(buffer,fmt)value;text=trim(adjustl(buffer))
      if(digits==0)then
        if(text(len(text):)=='.')text=text(:len(text)-1)
      end if
      if(len(text)>0)then
        if(text(1:1)=='.')text='0'//text
        if(len(text)>1)then
          if(text(:2)=='-.')text='-0'//text(2:)
        end if
      end if
    end if
  end function
  function support_text(value,weighted)result(text)
    real(dp),intent(in)::value
    logical,intent(in)::weighted
    character(:),allocatable::text
    character(40)::buffer
    if(ieee_is_nan(value))then
      text='nan'
    else if(.not.ieee_is_finite(value))then
      text='inf';if(value<0)text='-inf'
    else if(weighted)then
      text=float_text(value)
    else
      write(buffer,'(i0)')nint(value,int64);text=trim(buffer)
    end if
  end function
  subroutine render_report(report,digits,weighted,target_names)
    type(classification_report_result),intent(inout)::report
    integer,intent(in)::digits
    logical,intent(in)::weighted
    character(*),optional,intent(in)::target_names(:)
    integer::width,i
    character(:),allocatable::line
    call require(report%n_class_rows>0.or.present(target_names),'classification report text needs target names')
    width=max(12,digits)
    if(present(target_names))then
      call require(size(target_names)>0,'classification report text needs target names')
      do i=1,size(target_names)
        width=max(width,len_trim(target_names(i)))
      end do
    else
      do i=1,report%n_class_rows
        width=max(width,len(report%rows(i)%name))
      end do
    end if
    report%text=repeat(' ',width)//'  precision    recall  f1-score   support'//new_line('a')//new_line('a')
    do i=1,size(report%rows)
      if(i==report%n_class_rows+1)report%text=report%text//new_line('a')
      line=padded(report%rows(i)%name,width)//' '
      if(report%rows(i)%is_accuracy)then
        line=line//repeat(' ',20)
      else
        line=line//' '//padded(fixed(report%rows(i)%precision,digits),9)// &
          ' '//padded(fixed(report%rows(i)%recall,digits),9)
      end if
      line=line//' '//padded(fixed(report%rows(i)%f1_score,digits),9)// &
        ' '//padded(support_text(report%rows(i)%support,weighted),9)//new_line('a')
      report%text=report%text//line
    end do
  end subroutine
end module
