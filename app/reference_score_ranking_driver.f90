program reference_score_ranking_driver
  use sklearn_common, only: dp
  use sklearn_metrics, only: hinge_loss,top_k_accuracy_score,dcg_score,ndcg_score
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  integer, allocatable :: y(:),labels(:),cutoff
  real(dp), allocatable :: scores(:,:),relevance(:,:),weights(:),dcg_rows(:),ndcg_rows(:)
  integer :: cases,index,kind,n,p,known,k,weighted,ignore,nonnegative,integral,i
  real(dp) :: base,value,nan,inf
  character(60) :: mode
  nan=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
  call get_command_argument(1,mode)
  if(len_trim(mode)>0) then
    call reject(trim(mode));error stop 'invalid score ranking input accepted'
  end if
  read(*,*) cases
  do index=1,cases
    read(*,*) kind,n,p,known,k,weighted,ignore,nonnegative,integral,base
    allocate(scores(n,p))
    if(k>0) then
      allocate(cutoff);cutoff=k
    end if
    if(known>0) then
      allocate(labels(known));read(*,*) labels
    end if
    if(kind/=2) then
      allocate(y(n));read(*,*) y
      do i=1,n
        read(*,*) scores(i,:)
      end do
    else
      allocate(relevance(n,p),dcg_rows(n),ndcg_rows(n))
      do i=1,n
        read(*,*) relevance(i,:)
        read(*,*) scores(i,:)
      end do
    end if
    if(weighted==1) then
      allocate(weights(n));read(*,*) weights
    end if
    write(*,'(a,1x,i0)') 'BEGIN',index
    if(kind==2) then
      value=dcg_score(relevance,scores,k=cutoff,log_base=base,sample_weight=weights,ignore_ties=ignore==1)
      call vector('dcg',[value])
      do i=1,n
        dcg_rows(i)=dcg_score(relevance(i:i,:),scores(i:i,:),k=cutoff,log_base=base,ignore_ties=ignore==1)
      end do
      call vector('dcg_rows',dcg_rows)
      if(nonnegative==1) then
        value=ndcg_score(relevance,scores,k=cutoff,sample_weight=weights,ignore_ties=ignore==1)
        call vector('ndcg',[value])
        do i=1,n
          ndcg_rows(i)=ndcg_score(relevance(i:i,:),scores(i:i,:),k=cutoff,ignore_ties=ignore==1)
        end do
        call vector('ndcg_rows',ndcg_rows)
      end if
      value=dcg_score(relevance/=0,scores,k=cutoff,log_base=base,sample_weight=weights,ignore_ties=ignore==1)
      call vector('logical_dcg',[value])
      value=ndcg_score(relevance/=0,scores,k=cutoff,sample_weight=weights,ignore_ties=ignore==1)
      call vector('logical_ndcg',[value])
      if(integral==1) then
        value=dcg_score(int(relevance),scores,k=cutoff,log_base=base,sample_weight=weights,ignore_ties=ignore==1)
        call vector('integer_dcg',[value])
        if(nonnegative==1) then
          value=ndcg_score(int(relevance),scores,k=cutoff,sample_weight=weights,ignore_ties=ignore==1)
          call vector('integer_ndcg',[value])
        end if
      end if
    else
      if(kind/=4) then
        call vector('hinge',[hinge_loss(y,scores,labels,weights)])
        if(p==1) call vector('vector_hinge',[hinge_loss(y,scores(:,1),labels,weights)])
      end if
      if(kind/=3) then
        if(kind/=4) call vector('top_k',[top_k_accuracy_score(y,scores,k=cutoff,sample_weight=weights,labels=labels)])
        value=top_k_accuracy_score(y,scores,k=cutoff,normalize=.false.,sample_weight=weights,labels=labels)
        call vector('top_k_count',[value])
        if(p==1) then
          if(kind/=4) then
            value=top_k_accuracy_score(y,scores(:,1),k=cutoff,sample_weight=weights,labels=labels)
            call vector('vector_top_k',[value])
          end if
          value=top_k_accuracy_score(y,scores(:,1),k=cutoff,normalize=.false.,sample_weight=weights,labels=labels)
          call vector('vector_top_k_count',[value])
        end if
      end if
    end if
    write(*,'(a)') 'END'
    deallocate(scores)
    if(allocated(y)) deallocate(y)
    if(allocated(relevance)) deallocate(relevance,dcg_rows,ndcg_rows)
    if(allocated(labels)) deallocate(labels)
    if(allocated(weights)) deallocate(weights)
    if(allocated(cutoff)) deallocate(cutoff)
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
  subroutine reject(name)
    character(*), intent(in) :: name
    integer :: truth(3)
    real(dp) :: s(3,3),r(3,3),w(3)
    truth=[0,1,2];s=0;r=1;w=1
    select case(name)
    case('empty');value=hinge_loss([integer::],[real(dp)::])
    case('samples');value=hinge_loss(truth,s(:2,:))
    case('hinge_vector');value=hinge_loss(truth,s(:,1))
    case('hinge_columns');value=hinge_loss(truth,s(:,:2))
    case('hinge_binary');value=hinge_loss([0,1,0],s(:,:2))
    case('hinge_labels');value=hinge_loss(truth,s,labels=[0,1,7])
    case('top_k');value=top_k_accuracy_score(truth,s,k=0)
    case('top_labels');value=top_k_accuracy_score(truth,s,labels=[0,1,1])
    case('top_order');value=top_k_accuracy_score(truth,s,labels=[2,1,0])
    case('top_missing');value=top_k_accuracy_score(truth,s,labels=[0,1,7])
    case('top_columns');value=top_k_accuracy_score(truth,s(:,:2))
    case('top_binary_classes');value=top_k_accuracy_score([0,0,0],s(:,1))
    case('score_nan');s(1,1)=nan;value=hinge_loss(truth,s)
    case('score_inf');s(1,1)=inf;value=top_k_accuracy_score(truth,s)
    case('weight_length');value=hinge_loss(truth,s,sample_weight=w(:2))
    case('weight_nan');w(1)=nan;value=top_k_accuracy_score(truth,s,sample_weight=w)
    case('weight_inf');w(1)=inf;value=dcg_score(r,s,sample_weight=w)
    case('weight_zero');w=0;value=hinge_loss(truth,s,sample_weight=w)
    case('weight_cancel');w=[-1._dp,0._dp,1._dp];value=ndcg_score(r,s,sample_weight=w)
    case('documents');value=dcg_score(r(:,:1),s(:,:1))
    case('dcg_shape');value=dcg_score(r,s(:,:2))
    case('relevance_nan');r(1,1)=nan;value=ndcg_score(r,s)
    case('relevance_inf');r(1,1)=inf;value=dcg_score(r,s)
    case('negative_relevance');r(1,1)=-1;value=ndcg_score(r,s)
    case('dcg_k');value=dcg_score(r,s,k=0)
    case('base_zero');value=dcg_score(r,s,log_base=0._dp)
    case('base_negative');value=dcg_score(r,s,log_base=-1._dp)
    case('base_inf');value=dcg_score(r,s,log_base=inf)
    case('base_nan');value=dcg_score(r,s,log_base=nan)
    end select
  end subroutine
end program
