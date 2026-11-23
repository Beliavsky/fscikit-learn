program reference_tsne_bh_fit_driver
  use sklearn_common, only: dp,require
  use sklearn_tsne, only: tsne,trustworthiness
  use sklearn_sparse, only: csr_matrix,csr_from_dense
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  type(tsne) :: model
  type(csr_matrix) :: graph,original
  real(dp), allocatable :: x(:,:),initial(:,:),embedding(:,:),before(:,:),old(:,:),again(:,:)
  integer :: n,p,kind,init_mode,stored,metric,refit
  character(40) :: argument
  call get_command_argument(1,argument)
  if(len_trim(argument)>0) then
    call reject(trim(argument));error stop 'invalid Barnes-Hut fit accepted'
  end if
  read(*,*) kind,n,p,model%n_components,init_mode,model%max_iter,model%n_iter_without_progress,model%random_state, &
    metric,model%perplexity,model%early_exaggeration,model%learning_rate,model%min_grad_norm,model%angle,refit
  model%metric='euclidean'
  if(metric==1) model%metric='manhattan'
  if(metric==2) model%metric='cosine'
  if(metric==3) model%metric='precomputed'
  if(kind==1) then
    read(*,*) stored;graph%n_rows=n;graph%n_cols=p
    allocate(graph%indptr(n+1),graph%indices(stored),graph%data(stored))
    read(*,*) graph%indptr;read(*,*) graph%indices;read(*,*) graph%data;original=graph
  else
    allocate(x(n,p));read(*,*) x;before=x
  end if
  allocate(initial(n,model%n_components));read(*,*) initial;old=initial
  if(init_mode==1) model%init='random'
  if(init_mode==2) model%init='pca'
  call fit()
  embedding=model%embedding_
  call matrix(embedding)
  call integers(reshape(int([model%n_features_in_,model%n_iter_],int64),[2,1]))
  call matrix(reshape([model%kl_divergence_,model%learning_rate_],[2,1]))
  if(refit/=0) then
    call fit();again=model%embedding_
    call require(all(embedding==again),'Barnes-Hut seeded refit differs')
  end if
  if(kind==0) then
    call require(all(x==before),'Barnes-Hut fit input changed')
  else
    call require(all(graph%data==original%data).and.all(graph%indices==original%indices).and. &
      all(graph%indptr==original%indptr),'Barnes-Hut CSR input changed')
  end if
  call require(all(initial==old),'Barnes-Hut initialization changed')
  call integers(reshape([1_int64],[1,1]))
contains
  subroutine reject(name)
    character(*), intent(in) :: name
    real(dp) :: a(4,2),b(4,4)
    type(csr_matrix) :: g
    a=0;b=0;model%perplexity=2;model%max_iter=250;model%init='random'
    select case(name)
    case('dimensions');model%n_components=4
    case('angle_nan');model%angle=ieee_value(0._dp,ieee_quiet_nan)
    case('angle_negative');model%angle=-1
    case('angle_large');model%angle=2
    case('csr_exact');model%method='exact';model%metric='precomputed';g=csr_from_dense(b,.true.)
      call model%fit(g);return
    case('csr_metric');g=csr_from_dense(b,.true.);call model%fit(g);return
    case('csr_shape');model%metric='precomputed';g=csr_from_dense(a,.true.);call model%fit(g);return
    case('csr_neighbors');model%metric='precomputed';g=csr_from_dense(b);call model%fit(g);return
    case('csr_pca');model%metric='precomputed';model%init='pca';g=csr_from_dense(b,.true.)
      call model%fit(g);return
    case('csr_negative');model%metric='precomputed';b(1,2)=-1;g=csr_from_dense(b,.true.);call model%fit(g);return
    case default;error stop 'unknown Barnes-Hut fit rejection'
    end select
    call model%fit(a)
  end subroutine
  subroutine fit()
    if(kind==1) then
      if(init_mode==0) then
        call model%fit(graph,initial)
      else
        call model%fit(graph)
      end if
    else
      if(init_mode==0) then
        call model%fit(x,initial)
      else
        call model%fit(x)
      end if
    end if
  end subroutine
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    print *, 'F',shape(a,kind=int64)
    print '(8(es25.16e3,1x))',a
  end subroutine
  subroutine integers(a)
    integer(int64), intent(in) :: a(:,:)
    print *, 'I',shape(a,kind=int64)
    print '(16(i0,1x))',a
  end subroutine
end program
