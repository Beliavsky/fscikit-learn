program reference_classification_metrics_driver
  use sklearn_common, only: dp
  use sklearn_metrics, only: precision_score,recall_score,f1_score,jaccard_score,fbeta_score, &
    precision_recall_fscore_support,multilabel_confusion_matrix
  use sklearn_sparse, only: csr_matrix,csr_from_dense
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  integer, allocatable :: yv(:),pv(:),ym(:,:),pm(:,:),labels(:)
  real(dp), allocatable :: weights(:)
  type(csr_matrix) :: sy,sp
  integer :: cases,index,kind,n,q,avgcode,pos,weighted,haslabels,nlabels,jok,i
  real(dp) :: beta,zero,nan,inf,value
  logical :: boolean
  character(60) :: mode
  character(:), allocatable :: average
  nan=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
  call get_command_argument(1,mode)
  if(len_trim(mode)>0.and.trim(mode)/='jaccard_stdin') then
    call reject(trim(mode));error stop 'invalid classification metric input accepted'
  end if
  read(*,*) cases
  do index=1,cases
    read(*,*) kind,n,q,avgcode,pos,weighted,haslabels,nlabels,jok,beta,zero
    select case(avgcode)
    case(0);average='none'
    case(1);average='binary'
    case(2);average='micro'
    case(3);average='macro'
    case(4);average='weighted'
    case(5);average='samples'
    end select
    if(kind==0) then
      allocate(yv(n),pv(n));read(*,*) yv;read(*,*) pv
      boolean=all(yv==0.or.yv==1).and.all(pv==0.or.pv==1)
    else
      allocate(ym(n,q),pm(n,q))
      do i=1,n
        read(*,*) ym(i,:);read(*,*) pm(i,:)
      end do
      boolean=q>1.or.(all(ym==0.or.ym==1).and.all(pm==0.or.pm==1))
    end if
    if(weighted==1) then
      allocate(weights(n));read(*,*) weights
    end if
    if(haslabels==1) then
      allocate(labels(nlabels));if(nlabels>0) read(*,*) labels
    end if
    if(trim(mode)=='jaccard_stdin') then
      if(kind==0) then
        value=jaccard_score(yv,pv,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero)
      else
        value=jaccard_score(ym,pm,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero)
      end if
      error stop 'source Jaccard rejection accepted'
    end if
    write(*,'(a,1x,i0)') 'BEGIN',index
    if(kind==0) then
      call emit_vv(yv,pv,'int')
      if(boolean) call emit_bb(yv/=0,pv/=0,'logical')
    else
      call emit_ii(ym,pm,'int')
      if(boolean) call emit_ll(ym/=0,pm/=0,'logical')
      if(q>1) then
        sy=csr_from_dense(real(merge(1,0,ym/=0),dp));sp=csr_from_dense(real(merge(1,0,pm/=0),dp))
        call emit_ss(sy,sp,'csr');call emit_si(sy,pm,'csr_int');call emit_is(ym,sp,'int_csr')
        call emit_sl(sy,pm/=0,'csr_logical');call emit_ls(ym/=0,sp,'logical_csr')
      end if
    end if
    write(*,'(a)') 'END'
    if(allocated(yv)) deallocate(yv,pv)
    if(allocated(ym)) deallocate(ym,pm)
    if(allocated(weights)) deallocate(weights)
    if(allocated(labels)) deallocate(labels)
  end do
contains
  subroutine vector(name,x)
    character(*), intent(in) :: name
    real(dp), intent(in) :: x(:)
    integer :: j
    write(*,'(a,1x,i0)',advance='no') name,size(x)
    do j=1,size(x)
      write(*,'(1x,es25.17e3)',advance='no') x(j)
    end do
    write(*,*)
  end subroutine
  subroutine tensor(name,x)
    character(*), intent(in) :: name
    real(dp), intent(in) :: x(:,:,:)
    real(dp) :: flat(size(x))
    integer :: i,j,k,index
    index=0
    do i=1,size(x,1)
      do j=1,size(x,2)
        do k=1,size(x,3)
          index=index+1;flat(index)=x(i,j,k)
        end do
      end do
    end do
    call vector(name,flat)
  end subroutine
  subroutine emit_vv(y,p,prefix)
    integer, intent(in) :: y(:)
    integer, intent(in) :: p(:)
    character(*), intent(in) :: prefix
    real(dp), allocatable :: pr(:),re(:),fs(:),support(:),raw(:),cm(:,:,:)
    real(dp) :: v
    call precision_recall_fscore_support(y,p,pr,re,fs,support,beta=beta,labels=labels,pos_label=pos, &
      average=average,sample_weight=weights,zero_division=zero)
    call vector(prefix//'_summary_precision',pr);call vector(prefix//'_summary_recall',re)
    call vector(prefix//'_summary_fbeta',fs)
    call vector(prefix//'_support_present',[merge(1._dp,0._dp,allocated(support))])
    if(allocated(support)) then
      call vector(prefix//'_support',support)
    else
      call vector(prefix//'_support',[real(dp)::])
    end if
    cm=multilabel_confusion_matrix(y,p,sample_weight=weights,labels=labels)
    call tensor(prefix//'_confusion',cm)
    v=precision_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_precision',[v]);call vector(prefix//'_precision_raw',raw)
    v=recall_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_recall',[v]);call vector(prefix//'_recall_raw',raw)
    v=f1_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_f1',[v]);call vector(prefix//'_f1_raw',raw)
    v=fbeta_score(y,p,beta=beta,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_fbeta',[v]);call vector(prefix//'_fbeta_raw',raw)
    if(jok==1) then
      v=jaccard_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
      call vector(prefix//'_jaccard',[v]);call vector(prefix//'_jaccard_raw',raw)
    end if
  end subroutine
  subroutine emit_bb(y,p,prefix)
    logical, intent(in) :: y(:)
    logical, intent(in) :: p(:)
    character(*), intent(in) :: prefix
    real(dp), allocatable :: pr(:),re(:),fs(:),support(:),raw(:),cm(:,:,:)
    real(dp) :: v
    call precision_recall_fscore_support(y,p,pr,re,fs,support,beta=beta,labels=labels,pos_label=pos, &
      average=average,sample_weight=weights,zero_division=zero)
    call vector(prefix//'_summary_precision',pr);call vector(prefix//'_summary_recall',re)
    call vector(prefix//'_summary_fbeta',fs)
    call vector(prefix//'_support_present',[merge(1._dp,0._dp,allocated(support))])
    if(allocated(support)) then
      call vector(prefix//'_support',support)
    else
      call vector(prefix//'_support',[real(dp)::])
    end if
    cm=multilabel_confusion_matrix(y,p,sample_weight=weights,labels=labels)
    call tensor(prefix//'_confusion',cm)
    v=precision_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_precision',[v]);call vector(prefix//'_precision_raw',raw)
    v=recall_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_recall',[v]);call vector(prefix//'_recall_raw',raw)
    v=f1_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_f1',[v]);call vector(prefix//'_f1_raw',raw)
    v=fbeta_score(y,p,beta=beta,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_fbeta',[v]);call vector(prefix//'_fbeta_raw',raw)
    if(jok==1) then
      v=jaccard_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
      call vector(prefix//'_jaccard',[v]);call vector(prefix//'_jaccard_raw',raw)
    end if
  end subroutine
  subroutine emit_ii(y,p,prefix)
    integer, intent(in) :: y(:,:)
    integer, intent(in) :: p(:,:)
    character(*), intent(in) :: prefix
    real(dp), allocatable :: pr(:),re(:),fs(:),support(:),raw(:),cm(:,:,:)
    real(dp) :: v
    call precision_recall_fscore_support(y,p,pr,re,fs,support,beta=beta,labels=labels,pos_label=pos, &
      average=average,sample_weight=weights,zero_division=zero)
    call vector(prefix//'_summary_precision',pr);call vector(prefix//'_summary_recall',re)
    call vector(prefix//'_summary_fbeta',fs)
    call vector(prefix//'_support_present',[merge(1._dp,0._dp,allocated(support))])
    if(allocated(support)) then
      call vector(prefix//'_support',support)
    else
      call vector(prefix//'_support',[real(dp)::])
    end if
    cm=multilabel_confusion_matrix(y,p,sample_weight=weights,labels=labels)
    call tensor(prefix//'_confusion',cm)
    if(q>1) then
      cm=multilabel_confusion_matrix(y,p,sample_weight=weights,labels=labels,samplewise=.true.)
      call tensor(prefix//'_sample_confusion',cm)
    end if
    v=precision_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_precision',[v]);call vector(prefix//'_precision_raw',raw)
    v=recall_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_recall',[v]);call vector(prefix//'_recall_raw',raw)
    v=f1_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_f1',[v]);call vector(prefix//'_f1_raw',raw)
    v=fbeta_score(y,p,beta=beta,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_fbeta',[v]);call vector(prefix//'_fbeta_raw',raw)
    if(jok==1) then
      v=jaccard_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
      call vector(prefix//'_jaccard',[v]);call vector(prefix//'_jaccard_raw',raw)
    end if
  end subroutine
  subroutine emit_ll(y,p,prefix)
    logical, intent(in) :: y(:,:)
    logical, intent(in) :: p(:,:)
    character(*), intent(in) :: prefix
    real(dp), allocatable :: pr(:),re(:),fs(:),support(:),raw(:),cm(:,:,:)
    real(dp) :: v
    call precision_recall_fscore_support(y,p,pr,re,fs,support,beta=beta,labels=labels,pos_label=pos, &
      average=average,sample_weight=weights,zero_division=zero)
    call vector(prefix//'_summary_precision',pr);call vector(prefix//'_summary_recall',re)
    call vector(prefix//'_summary_fbeta',fs)
    call vector(prefix//'_support_present',[merge(1._dp,0._dp,allocated(support))])
    if(allocated(support)) then
      call vector(prefix//'_support',support)
    else
      call vector(prefix//'_support',[real(dp)::])
    end if
    cm=multilabel_confusion_matrix(y,p,sample_weight=weights,labels=labels)
    call tensor(prefix//'_confusion',cm)
    if(q>1) then
      cm=multilabel_confusion_matrix(y,p,sample_weight=weights,labels=labels,samplewise=.true.)
      call tensor(prefix//'_sample_confusion',cm)
    end if
    v=precision_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_precision',[v]);call vector(prefix//'_precision_raw',raw)
    v=recall_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_recall',[v]);call vector(prefix//'_recall_raw',raw)
    v=f1_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_f1',[v]);call vector(prefix//'_f1_raw',raw)
    v=fbeta_score(y,p,beta=beta,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_fbeta',[v]);call vector(prefix//'_fbeta_raw',raw)
    if(jok==1) then
      v=jaccard_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
      call vector(prefix//'_jaccard',[v]);call vector(prefix//'_jaccard_raw',raw)
    end if
  end subroutine
  subroutine emit_ss(y,p,prefix)
    type(csr_matrix), intent(in) :: y
    type(csr_matrix), intent(in) :: p
    character(*), intent(in) :: prefix
    real(dp), allocatable :: pr(:),re(:),fs(:),support(:),raw(:),cm(:,:,:)
    real(dp) :: v
    call precision_recall_fscore_support(y,p,pr,re,fs,support,beta=beta,labels=labels,pos_label=pos, &
      average=average,sample_weight=weights,zero_division=zero)
    call vector(prefix//'_summary_precision',pr);call vector(prefix//'_summary_recall',re)
    call vector(prefix//'_summary_fbeta',fs)
    call vector(prefix//'_support_present',[merge(1._dp,0._dp,allocated(support))])
    if(allocated(support)) then
      call vector(prefix//'_support',support)
    else
      call vector(prefix//'_support',[real(dp)::])
    end if
    cm=multilabel_confusion_matrix(y,p,sample_weight=weights,labels=labels)
    call tensor(prefix//'_confusion',cm)
    if(q>1) then
      cm=multilabel_confusion_matrix(y,p,sample_weight=weights,labels=labels,samplewise=.true.)
      call tensor(prefix//'_sample_confusion',cm)
    end if
    v=precision_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_precision',[v]);call vector(prefix//'_precision_raw',raw)
    v=recall_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_recall',[v]);call vector(prefix//'_recall_raw',raw)
    v=f1_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_f1',[v]);call vector(prefix//'_f1_raw',raw)
    v=fbeta_score(y,p,beta=beta,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_fbeta',[v]);call vector(prefix//'_fbeta_raw',raw)
    if(jok==1) then
      v=jaccard_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
      call vector(prefix//'_jaccard',[v]);call vector(prefix//'_jaccard_raw',raw)
    end if
  end subroutine
  subroutine emit_si(y,p,prefix)
    type(csr_matrix), intent(in) :: y
    integer, intent(in) :: p(:,:)
    character(*), intent(in) :: prefix
    real(dp), allocatable :: pr(:),re(:),fs(:),support(:),raw(:),cm(:,:,:)
    real(dp) :: v
    call precision_recall_fscore_support(y,p,pr,re,fs,support,beta=beta,labels=labels,pos_label=pos, &
      average=average,sample_weight=weights,zero_division=zero)
    call vector(prefix//'_summary_precision',pr);call vector(prefix//'_summary_recall',re)
    call vector(prefix//'_summary_fbeta',fs)
    call vector(prefix//'_support_present',[merge(1._dp,0._dp,allocated(support))])
    if(allocated(support)) then
      call vector(prefix//'_support',support)
    else
      call vector(prefix//'_support',[real(dp)::])
    end if
    cm=multilabel_confusion_matrix(y,p,sample_weight=weights,labels=labels)
    call tensor(prefix//'_confusion',cm)
    if(q>1) then
      cm=multilabel_confusion_matrix(y,p,sample_weight=weights,labels=labels,samplewise=.true.)
      call tensor(prefix//'_sample_confusion',cm)
    end if
    v=precision_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_precision',[v]);call vector(prefix//'_precision_raw',raw)
    v=recall_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_recall',[v]);call vector(prefix//'_recall_raw',raw)
    v=f1_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_f1',[v]);call vector(prefix//'_f1_raw',raw)
    v=fbeta_score(y,p,beta=beta,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_fbeta',[v]);call vector(prefix//'_fbeta_raw',raw)
    if(jok==1) then
      v=jaccard_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
      call vector(prefix//'_jaccard',[v]);call vector(prefix//'_jaccard_raw',raw)
    end if
  end subroutine
  subroutine emit_is(y,p,prefix)
    integer, intent(in) :: y(:,:)
    type(csr_matrix), intent(in) :: p
    character(*), intent(in) :: prefix
    real(dp), allocatable :: pr(:),re(:),fs(:),support(:),raw(:),cm(:,:,:)
    real(dp) :: v
    call precision_recall_fscore_support(y,p,pr,re,fs,support,beta=beta,labels=labels,pos_label=pos, &
      average=average,sample_weight=weights,zero_division=zero)
    call vector(prefix//'_summary_precision',pr);call vector(prefix//'_summary_recall',re)
    call vector(prefix//'_summary_fbeta',fs)
    call vector(prefix//'_support_present',[merge(1._dp,0._dp,allocated(support))])
    if(allocated(support)) then
      call vector(prefix//'_support',support)
    else
      call vector(prefix//'_support',[real(dp)::])
    end if
    cm=multilabel_confusion_matrix(y,p,sample_weight=weights,labels=labels)
    call tensor(prefix//'_confusion',cm)
    if(q>1) then
      cm=multilabel_confusion_matrix(y,p,sample_weight=weights,labels=labels,samplewise=.true.)
      call tensor(prefix//'_sample_confusion',cm)
    end if
    v=precision_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_precision',[v]);call vector(prefix//'_precision_raw',raw)
    v=recall_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_recall',[v]);call vector(prefix//'_recall_raw',raw)
    v=f1_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_f1',[v]);call vector(prefix//'_f1_raw',raw)
    v=fbeta_score(y,p,beta=beta,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_fbeta',[v]);call vector(prefix//'_fbeta_raw',raw)
    if(jok==1) then
      v=jaccard_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
      call vector(prefix//'_jaccard',[v]);call vector(prefix//'_jaccard_raw',raw)
    end if
  end subroutine
  subroutine emit_sl(y,p,prefix)
    type(csr_matrix), intent(in) :: y
    logical, intent(in) :: p(:,:)
    character(*), intent(in) :: prefix
    real(dp), allocatable :: pr(:),re(:),fs(:),support(:),raw(:),cm(:,:,:)
    real(dp) :: v
    call precision_recall_fscore_support(y,p,pr,re,fs,support,beta=beta,labels=labels,pos_label=pos, &
      average=average,sample_weight=weights,zero_division=zero)
    call vector(prefix//'_summary_precision',pr);call vector(prefix//'_summary_recall',re)
    call vector(prefix//'_summary_fbeta',fs)
    call vector(prefix//'_support_present',[merge(1._dp,0._dp,allocated(support))])
    if(allocated(support)) then
      call vector(prefix//'_support',support)
    else
      call vector(prefix//'_support',[real(dp)::])
    end if
    cm=multilabel_confusion_matrix(y,p,sample_weight=weights,labels=labels)
    call tensor(prefix//'_confusion',cm)
    if(q>1) then
      cm=multilabel_confusion_matrix(y,p,sample_weight=weights,labels=labels,samplewise=.true.)
      call tensor(prefix//'_sample_confusion',cm)
    end if
    v=precision_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_precision',[v]);call vector(prefix//'_precision_raw',raw)
    v=recall_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_recall',[v]);call vector(prefix//'_recall_raw',raw)
    v=f1_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_f1',[v]);call vector(prefix//'_f1_raw',raw)
    v=fbeta_score(y,p,beta=beta,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_fbeta',[v]);call vector(prefix//'_fbeta_raw',raw)
    if(jok==1) then
      v=jaccard_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
      call vector(prefix//'_jaccard',[v]);call vector(prefix//'_jaccard_raw',raw)
    end if
  end subroutine
  subroutine emit_ls(y,p,prefix)
    logical, intent(in) :: y(:,:)
    type(csr_matrix), intent(in) :: p
    character(*), intent(in) :: prefix
    real(dp), allocatable :: pr(:),re(:),fs(:),support(:),raw(:),cm(:,:,:)
    real(dp) :: v
    call precision_recall_fscore_support(y,p,pr,re,fs,support,beta=beta,labels=labels,pos_label=pos, &
      average=average,sample_weight=weights,zero_division=zero)
    call vector(prefix//'_summary_precision',pr);call vector(prefix//'_summary_recall',re)
    call vector(prefix//'_summary_fbeta',fs)
    call vector(prefix//'_support_present',[merge(1._dp,0._dp,allocated(support))])
    if(allocated(support)) then
      call vector(prefix//'_support',support)
    else
      call vector(prefix//'_support',[real(dp)::])
    end if
    cm=multilabel_confusion_matrix(y,p,sample_weight=weights,labels=labels)
    call tensor(prefix//'_confusion',cm)
    if(q>1) then
      cm=multilabel_confusion_matrix(y,p,sample_weight=weights,labels=labels,samplewise=.true.)
      call tensor(prefix//'_sample_confusion',cm)
    end if
    v=precision_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_precision',[v]);call vector(prefix//'_precision_raw',raw)
    v=recall_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_recall',[v]);call vector(prefix//'_recall_raw',raw)
    v=f1_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_f1',[v]);call vector(prefix//'_f1_raw',raw)
    v=fbeta_score(y,p,beta=beta,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
    call vector(prefix//'_fbeta',[v]);call vector(prefix//'_fbeta_raw',raw)
    if(jok==1) then
      v=jaccard_score(y,p,average=average,pos_label=pos,sample_weight=weights,labels=labels,zero_division=zero,raw_values=raw)
      call vector(prefix//'_jaccard',[v]);call vector(prefix//'_jaccard_raw',raw)
    end if
  end subroutine
  subroutine reject(name)
    character(*), intent(in) :: name
    integer :: y(3),p(3),a(2,3),b(2,3)
    real(dp), allocatable :: pr(:),re(:),fs(:),support(:),cm(:,:,:)
    real(dp) :: w(3)
    type(csr_matrix) :: sparse
    y=[0,1,1];p=[0,0,1];a=0;b=0;a(:,1)=1;b(:,2)=1;w=1
    sparse=csr_from_dense(real(a,dp))
    select case(name)
    case('length');value=f1_score(y,p(:2),average='micro')
    case('shape');value=precision_score(a,b(:,:2),average='macro')
    case('matrix_empty');value=recall_score(reshape([integer::],[0,3]),reshape([integer::],[0,3]),average='macro')
    case('average');value=f1_score(y,p,average='invalid')
    case('multiclass_binary');value=f1_score([0,1,2],[0,1,2])
    case('multilabel_binary');value=precision_score(a,b)
    case('pos_label');value=recall_score(y,p,pos_label=7)
    case('samples_vector');value=f1_score(y,p,average='samples')
    case('labels_high');value=f1_score(a,b,average='macro',labels=[3])
    case('labels_low');value=precision_score(a,b,average='macro',labels=[-1])
    case('indicator');a(1,:)=[0,1,2];value=f1_score(a,b,average='macro')
    case('weight_length');value=f1_score(y,p,sample_weight=w(:2))
    case('zero_division');value=f1_score(y,p,zero_division=2._dp)
    case('jaccard_nan');value=jaccard_score(y,p,zero_division=nan)
    case('beta_negative');value=fbeta_score(y,p,-1._dp)
    case('beta_nan');value=fbeta_score(y,p,nan)
    case('beta_overflow');value=fbeta_score(y,p,1.e308_dp)
    case('jaccard_zero_weights');w=0;value=jaccard_score(a,b,average='samples',sample_weight=w(:2))
    case('csr_single');sparse=csr_from_dense(reshape([0._dp,1._dp],[2,1]));value=f1_score(sparse,sparse,average='macro')
    case('csr_value');sparse%data(1)=2;value=f1_score(sparse,sparse,average='macro')
    case('csr_nan');sparse%data(1)=nan;value=precision_score(sparse,sparse,average='micro')
    case('csr_index');sparse%indices(1)=4;value=f1_score(sparse,sparse,average='macro')
    case('csr_duplicate')
      sparse%n_rows=2;sparse%n_cols=3;sparse%indptr=[1,3,3];sparse%indices=[1,1];sparse%data=[1._dp,1._dp]
      value=f1_score(sparse,sparse,average='macro')
    case('csr_unsorted')
      sparse%n_rows=2;sparse%n_cols=3;sparse%indptr=[1,3,3];sparse%indices=[3,1];sparse%data=[1._dp,1._dp]
      value=jaccard_score(sparse,sparse,average='micro')
    end select
  end subroutine
end program
