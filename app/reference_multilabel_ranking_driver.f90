program reference_multilabel_ranking_driver
  use sklearn_common, only: dp
  use sklearn_metrics, only: coverage_error,label_ranking_average_precision_score,label_ranking_loss
  use sklearn_sparse, only: csr_matrix,csr_from_dense
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  integer, allocatable :: y(:,:)
  real(dp), allocatable :: scores(:,:),weights(:),c(:),a(:),loss(:)
  type(csr_matrix) :: csr
  integer :: cases,index,n,p,weighted,sparse,i
  character(64) :: mode
  real(dp) :: value,nan,inf
  nan=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
  call get_command_argument(1,mode)
  if(len_trim(mode)>0) then
    call reject(trim(mode));error stop 'invalid multilabel ranking input accepted'
  end if
  read(*,*) cases
  do index=1,cases
    read(*,*) n,p,weighted,sparse
    allocate(y(n,p),scores(n,p),c(n),a(n),loss(n))
    do i=1,n
      read(*,*) y(i,:)
      read(*,*) scores(i,:)
    end do
    if(weighted==1) then
      allocate(weights(n));read(*,*) weights
    end if
    write(*,'(a,1x,i0)') 'BEGIN',index
    if(p>1) then
      call vector('coverage',[coverage_error(y,scores,weights)])
      call vector('loss',[label_ranking_loss(y,scores,weights)])
      call vector('logical_coverage',[coverage_error(y/=0,scores,weights)])
      call vector('logical_loss',[label_ranking_loss(y/=0,scores,weights)])
      do i=1,n
        c(i)=coverage_error(y(i:i,:),scores(i:i,:))
        loss(i)=label_ranking_loss(y(i:i,:),scores(i:i,:))
      end do
      call vector('row_coverage',c);call vector('row_loss',loss)
    end if
    call vector('precision',[label_ranking_average_precision_score(y,scores,weights)])
    call vector('logical_precision',[label_ranking_average_precision_score(y/=0,scores,weights)])
    do i=1,n
      a(i)=label_ranking_average_precision_score(y(i:i,:),scores(i:i,:))
    end do
    call vector('row_precision',a)
    if(sparse==1) then
      csr=csr_from_dense(real(y,dp))
      call vector('csr_precision',[label_ranking_average_precision_score(csr,scores,weights)])
      if(p>1) call vector('csr_loss',[label_ranking_loss(csr,scores,weights)])
    end if
    write(*,'(a)') 'END'
    deallocate(y,scores,c,a,loss)
    if(allocated(weights)) deallocate(weights)
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
    integer :: labels(2,3)
    real(dp) :: s(2,3),w(2)
    labels=0;labels(:,1)=1;s=0;w=1
    select case(name)
    case('empty');value=coverage_error(reshape([integer::],[0,3]),reshape([real(dp)::],[0,3]))
    case('no_labels');value=label_ranking_average_precision_score(reshape([integer::],[2,0]),reshape([real(dp)::],[2,0]))
    case('single_coverage');value=coverage_error(labels(:,:1),s(:,:1))
    case('single_loss');value=label_ranking_loss(labels(:,:1),s(:,:1))
    case('shape');value=coverage_error(labels,s(:,:2))
    case('target');labels(1,:)=[0,1,2];value=label_ranking_loss(labels,s)
    case('nan');s(1,1)=nan;value=coverage_error(labels,s)
    case('inf');s(1,1)=inf;value=label_ranking_average_precision_score(labels,s)
    case('weight_length');value=label_ranking_loss(labels,s,w(:1))
    case('weight_nan');w(1)=nan;value=coverage_error(labels,s,w)
    case('weight_inf');w(1)=inf;value=label_ranking_loss(labels,s,w)
    case('weight_zero');w=0;value=coverage_error(labels,s,w)
    case('weight_cancel');w=[-1._dp,1._dp];value=label_ranking_average_precision_score(labels,s,w)
    case('csr_value');csr=csr_from_dense(real(labels,dp));csr%data(1)=0;value=label_ranking_loss(csr,s)
    case('csr_duplicate')
      csr%n_rows=2;csr%n_cols=3;csr%indptr=[1,3,3];csr%indices=[1,1];csr%data=[1._dp,1._dp]
      value=label_ranking_average_precision_score(csr,s)
    case('csr_unsorted')
      csr%n_rows=2;csr%n_cols=3;csr%indptr=[1,3,3];csr%indices=[3,1];csr%data=[1._dp,1._dp]
      value=label_ranking_loss(csr,s)
    case('csr_shape');csr=csr_from_dense(real(labels,dp));value=label_ranking_loss(csr,s(:,:2))
    case('csr_invalid');csr=csr_from_dense(real(labels,dp));csr%indices(1)=4;value=label_ranking_loss(csr,s)
    end select
  end subroutine
end program
